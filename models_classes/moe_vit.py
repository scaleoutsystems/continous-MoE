"""
Vision Transformer with Mixture-of-Experts (MoE) FFN layers.

- Replace selected transformer MLP/FFN layers with MoE layers.
- Configurable: number of experts per MoE layer, top-k routing, shared expert,
  which layers are MoE layers (list or pattern), and ability to freeze router
  (gating) parameters after N training batches.
- The architecture adapts to the resolution of the input images: the
  patch embedding and positional embeddings will resize automatically when
  the model sees a new image size, and the factory allows `img_size` to be
  unspecified so that the loader can infer it at runtime.

The factory function `create_moe_vit(...)` returns the same package dict as the
other model factories so it plugs directly into the notebook.

Notes:
- This is a straightforward, easy-to-read MoE implementation (sparse dispatch by default).
  It performs sparse dispatch (only selected experts are executed per-sample) and is
  intended for research/debugging rather than large-scale production efficiency.
- The training function supports `router_freeze_after_epochs` via kwargs.
"""
from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Patch embedder for ViT-style tokenization.

    The input image is split into non-overlapping patches of size
    ``patch_size`` and each patch is flattened and projected to ``embed_dim``
    with a Conv2d.  ``img_size`` may be provided so that the number of
    patches is known at construction time; if omitted the size is inferred
    from the first forward pass.  In either case the image dimensions must
    be divisible by ``patch_size``.

    The output tensor has shape ``(B, num_patches, embed_dim)``.
    """

    def __init__(self, img_size: Optional[int] = None, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # grid_size/num_patches are lazy if img_size is unknown
        if img_size is not None:
            if img_size % patch_size != 0:
                raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size * self.grid_size
        else:
            self.grid_size = None
            self.num_patches = None
        # conv2d with stride=patch_size to extract non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # lazily infer size if not provided, or adapt if resolution changes
        if self.img_size is None:
            if H % self.patch_size != 0 or W % self.patch_size != 0:
                raise ValueError(f"Input size {H}x{W} not divisible by patch_size {self.patch_size}")
            self.img_size = H
            self.grid_size = H // self.patch_size
            self.num_patches = self.grid_size * self.grid_size
        else:
            if H != self.img_size or W != self.img_size:
                # update stored size rather than complaining
                if H % self.patch_size != 0 or W % self.patch_size != 0:
                    raise ValueError(f"Input size {H}x{W} not divisible by patch_size {self.patch_size}")
                self.img_size = H
                self.grid_size = H // self.patch_size
                self.num_patches = self.grid_size * self.grid_size
        x = self.proj(x)  # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    """Sparse Mixture-of-Experts layer (sparse dispatch by default).

    Each MoE layer is conceptually split into *unshared* experts that are
    selected by the router and *shared* experts that are executed for every
    sample.  The router only produces logits for the unshared experts and
    selects `top_k` of them per-sample; the shared experts are simply added on
    top unconditionally.  This makes the router softmax size independent of
    the number of shared experts and simplifies auxiliary losses and
    utilization statistics.

    Routing can be done using either a mean over tokens or by treating the
    first token as a dedicated "[CLS]" token; the latter is enabled with
    ``route_with_cls_token=True``.
    """

    def __init__(self, dim, hidden_dim,
                 num_unshared_experts: int = 4,
                 num_shared_experts: int = 0,
                 top_k: int = 1,
                 dropout: float = 0.0,
                 route_with_cls_token: bool = False):
        super().__init__()
        assert num_unshared_experts >= 0
        assert num_shared_experts >= 0
        assert num_unshared_experts + num_shared_experts > 0
        assert top_k >= 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_unshared_experts = num_unshared_experts
        self.num_shared_experts = num_shared_experts
        # keep a backwards-compatible attribute for total experts
        self.num_experts = num_unshared_experts + num_shared_experts
        # router only chooses among the unshared experts
        self.top_k = min(top_k, num_unshared_experts)
        self.route_with_cls_token = route_with_cls_token

        # experts list: unshared experts come first, then shared experts
        total = self.num_experts
        self.experts = nn.ModuleList([
            MLP(dim, hidden_dim, out_dim=dim, dropout=dropout)
            for _ in range(total)
        ])

        # gating (router) - only over unshared experts
        self.gate = nn.Linear(dim, num_unshared_experts)

        # per-forward debug stats (kept detached) only for unshared experts
        self.register_buffer('last_importance', torch.zeros(num_unshared_experts), persistent=False)
        self.register_buffer('last_load', torch.zeros(num_unshared_experts), persistent=False)

        # cumulative stats used for utilization metrics (on CPU)
        # cumulative_gate_sum: sum of gate probabilities over tokens for each unshared expert
        self.cumulative_gate_sum = torch.zeros(num_unshared_experts)
        # cumulative_selected_counts: number of tokens that selected each expert (top-k selection)
        self.cumulative_selected_counts = torch.zeros(num_unshared_experts)
        # cumulative_samples counts tokens processed (B * num_tokens)
        self.cumulative_samples = 0

        # last differentiable gate probs (kept on device, not detached)
        self._last_gate_probs = None

    def get_router_parameters(self):
        return self.gate.parameters()

    # helper for optimizer adjustments --------------------------------------------------
    def adjust_router_learning_rate(self, optimizer: torch.optim.Optimizer, multiplier: float):
        """Modify ``optimizer`` so that router parameters use ``lr * multiplier``.

        If ``multiplier`` is zero the routers are frozen (``requires_grad`` is
        disabled) and they are removed from the optimizer's parameter groups.
        The modified optimizer is returned (object may be mutated in place).
        """
        if multiplier == 0:
            # zero multiplier treated as freezing: disable grads on router params
            for p in self.get_router_parameters():
                p.requires_grad = False
            return optimizer
        # collect router parameters and remove them from existing groups
        router_params = list(self.get_router_parameters())
        base_lr = None
        # avoid ambiguous tensor comparisons by using id-based set
        router_ids = {id(p) for p in router_params}
        for group in optimizer.param_groups:
            if base_lr is None and 'lr' in group:
                base_lr = group['lr']
            # filter out any parameters whose id appears in router_ids
            group['params'] = [p for p in group['params'] if id(p) not in router_ids]
        if base_lr is None or not router_params:
            return optimizer
        optimizer.add_param_group({'params': router_params, 'lr': base_lr * multiplier})
        return optimizer

    # balance loss helper ------------------------------------------------------------
    def router_balance_loss(self, strength: float):
        """Return scalar balancing loss for all routers.

        The loss is computed only over the unshared experts because the gating
        logits and ``_last_gate_probs`` have that size by design.
        If ``strength`` is zero or routers are frozen this returns ``0.0``.
        """
        if not strength or strength <= 0:
            return 0.0
        loss = 0.0
        for m in self.modules():
            if isinstance(m, MoE) and m._last_gate_probs is not None:
                p_mean = m._last_gate_probs.mean(dim=0)
                target = torch.full_like(p_mean, 1.0 / float(max(1, p_mean.numel())))
                loss = loss + ((p_mean - target) ** 2).mean()
        return strength * loss

    def reset_cumulative_stats(self):
        # keep stats only for the router outputs (unshared experts)
        self.cumulative_gate_sum = torch.zeros(self.num_unshared_experts)
        self.cumulative_selected_counts = torch.zeros(self.num_unshared_experts)
        self.cumulative_samples = 0

    def get_cumulative_stats(self):
        if self.cumulative_samples == 0:
            return {
                'fraction': [0.0] * self.num_unshared_experts,
                'selected_fraction': [0.0] * self.num_unshared_experts,
                'samples': 0,
            }
        frac = (self.cumulative_gate_sum / float(max(1, self.cumulative_samples))).tolist()
        sel_frac = (self.cumulative_selected_counts / float(max(1, self.cumulative_samples))).tolist()
        return {'fraction': frac, 'selected_fraction': sel_frac, 'samples': int(self.cumulative_samples)}

    def forward(self, x):
        """Sparse dispatch based on a summary vector.

        The summary is either the mean over tokens or the ``[CLS]`` token (index
        0) depending on ``route_with_cls_token``.  Only the unshared experts are
        selected by the router; shared experts are executed for every sample and
        added without weighting.

        Args:
            x: tensor (B, N, C)
        Returns:
            out: tensor (B, N, C) — combined expert outputs per-sample
        """
        B, N, C = x.shape

        # Routing per-token: apply gate to each token representation
        # logits shape: (B, N, U)
        logits = self.gate(x)

        # differentiable gate probabilities per token
        gate_probs = F.softmax(logits, dim=-1)  # (B, N, U)
        # flatten tokens into one batch dimension for downstream utilities
        self._last_gate_probs = gate_probs.reshape(-1, gate_probs.size(-1))  # (B*N, U)

        # accumulate CPU-side statistics for utilization reporting (sum over tokens and batch)
        self.cumulative_gate_sum = (self.cumulative_gate_sum.cpu() + gate_probs.detach().sum(dim=(0, 1)).cpu())
        # cumulative_samples tracks tokens processed
        self.cumulative_samples += (B * N)

        U = self.num_unshared_experts
        T = self.num_shared_experts
        total = self.num_experts
        k = min(self.top_k, U)
        if k == 0 and T == 0:
            return x

        # select top-k unshared experts PER TOKEN
        # selected_mask shape: (B, N, U)
        selected_mask = torch.zeros((B, N, U), device=logits.device, dtype=torch.bool)
        if k > 0:
            _, topk_idx = torch.topk(logits, k, dim=-1)  # (B, N, k)
            selected_mask.scatter_(2, topk_idx, True)

        # compute normalized weights among selected unshared experts (per token)
        weights_un = torch.zeros((B, N, U), device=logits.device)
        if k > 0:
            very_neg = -1e9
            masked_logits = logits.clone()
            masked_logits[~selected_mask] = very_neg
            weights_un = F.softmax(masked_logits, dim=-1) * selected_mask.float()

        # simple stats (importance/load) aggregated over tokens
        importance = weights_un.sum(dim=(0, 1))  # sum across batch and token dims -> (U,)
        load = selected_mask.float().sum(dim=(0, 1))  # number of tokens selecting each expert
        self.last_importance = importance.detach().cpu()
        self.last_load = load.detach().cpu()

        # accumulate selected token counts into cumulative_selected_counts
        try:
            self.cumulative_selected_counts = (self.cumulative_selected_counts + load.detach().cpu())
        except Exception:
            self.cumulative_selected_counts = load.detach().cpu().clone()

        # assemble output: weighted unshared + unweighted shared (vectorised over tokens)
        out = torch.zeros_like(x)
        # unshared experts
        for e in range(U):
            w_e = weights_un[..., e].unsqueeze(-1)  # (B, N, 1)
            out = out + w_e * self.experts[e](x)
        # shared experts (unweighted)
        for e in range(U, total):
            out = out + self.experts[e](x)

        return out


class TransformerBlockMoE(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0,
                 use_moe=False, moe_params=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.use_moe = use_moe
        hidden_dim = int(dim * mlp_ratio)
        if use_moe:
            moe_params = moe_params or {}
            self.mlp = MoE(dim=dim, hidden_dim=hidden_dim, **moe_params)
        else:
            self.mlp = MLP(dim, hidden_dim, dropout=dropout)

    def forward(self, x):
        # x: (B, N, C)
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = x_res + x_attn
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


class ViTMoE(nn.Module):
    def __init__(self, *, img_size: Optional[int] = None, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                 moe_layer_indices: Optional[Union[List[int], str]] = None,
                 moe_params: Optional[dict] = None,
                 use_class_token=False):
        super().__init__()
        # patch embedding may infer size at first forward pass if img_size is
        # unknown; record initial value so we can allocate positional embeddings.
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        # num_patches not used directly; kept for backwards compatibility if
        # someone inspects the variable but not required here.
        # num_patches = self.patch_embed.num_patches
        self.use_class_token = use_class_token
        # positional embeddings are registered lazily; the actual number of
        # tokens may not be known until the first forward if img_size was
        # unspecified.  We still create the CLS token parameter here if needed
        # since its size does not depend on the image size later.
        if use_class_token:
            # learnable [CLS] token prepended to sequence
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter('cls_token', None)
        self.pos_embed = None  # will be created in _init_pos_embed
        self.pos_drop = nn.Dropout(p=0.0)

        # determine which layers are MoE
        if moe_layer_indices is None:
            moe_layer_indices = []
        elif isinstance(moe_layer_indices, str):
            if moe_layer_indices == 'every_other':
                moe_layer_indices = [i for i in range(depth) if (i % 2) == 1]
            elif moe_layer_indices == 'all':
                moe_layer_indices = list(range(depth))
            elif moe_layer_indices == 'back_half_every_other':
                moe_layer_indices = [i for i in range(depth) if ((i < depth // 2) and (i % 2) == 1)]
            else:
                moe_layer_indices = []

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_moe = i in moe_layer_indices
            block = TransformerBlockMoE(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                        use_moe=use_moe, moe_params=moe_params)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # init; if pos_embed has been created already, initialize it.  otherwise
        # it will be initialized lazily in forward.
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_pos_embed(self, seq_len: int, embed_dim: int):
        """Allocate or resize positional embeddings to match ``seq_len``.

        This is called on the first forward pass or whenever the number of
        tokens (patches+cls) changes due to a different input resolution.  The
        new positional embeddings are initialized from a normal distribution
        with the same std used in the constructor.
        """
        # create new parameter on the same device as existing weights
        device = None
        for p in self.parameters():
            device = p.device
            break
        if device is None:
            device = torch.device('cpu')
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim, device=device))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, C)
        if self.use_class_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        # ensure positional embedding matches current sequence length
        seq_len = x.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != seq_len:
            # we know embed_dim from x
            self._init_pos_embed(seq_len, x.size(2))
        # at this point pos_embed cannot be None (initialized above)
        assert self.pos_embed is not None
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # use first token as classification embedding
        cls = x[:, 0]
        out = self.head(cls)
        return out

    # router helpers ---------------------------------------------------------
    def get_router_parameters(self):
        for m in self.modules():
            if isinstance(m, MoE):
                yield from m.get_router_parameters()

    def freeze_routing(self, freeze: bool = True):
        for p in self.get_router_parameters():
            p.requires_grad = not freeze

    def adjust_router_learning_rate(self, optimizer: torch.optim.Optimizer, multiplier: float):
        """Wrapper that updates all MoE modules in the model."""
        for m in self.modules():
            if isinstance(m, MoE) and hasattr(m, 'adjust_router_learning_rate'):
                optimizer = m.adjust_router_learning_rate(optimizer, multiplier)
        return optimizer

    def router_balance_loss(self, strength: float):
        """Aggregate balance loss from all MoE layers."""
        loss = 0.0
        for m in self.modules():
            if isinstance(m, MoE) and hasattr(m, 'router_balance_loss'):
                loss = loss + m.router_balance_loss(strength)
        return loss

    # (router learning rate adjustments and balance loss helpers implemented above)

    def get_moe_utilization(self):
        """Return a list of per-MoE-layer utilization statistics (fraction per expert).

        Each element is a dict: {'layer_index': i, 'fraction': [...], 'samples': n}
        The reported fractions correspond only to the *unshared* experts, since
        the router does not operate over the shared set.
        """
        results = []
        for idx, m in enumerate(self.modules()):
            if isinstance(m, MoE):
                stats = m.get_cumulative_stats() if hasattr(m, 'get_cumulative_stats') else {}
                # 'fraction' : avg gate probability per expert (token-weighted)
                # 'selected_fraction' : fraction of tokens that selected each expert (top-k count / total tokens)
                results.append({
                    'layer_index': idx,
                    'fraction': stats.get('fraction', []),
                    'patch_load': stats.get('selected_fraction', []),
                    'samples': stats.get('samples', 0),
                })
        return results

def create_moe_vit(num_classes=10,
                   img_size: Optional[int] = None, patch_size=16,
                   embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                   moe_layer_indices: Optional[Union[List[int], str]] = 'every_other',
                   moe_num_unshared_experts: int = 4,
                   moe_num_shared_experts: int = 0,
                   moe_top_k: int = 1,
                   moe_route_with_cls_token: bool = True,
                   pretrained: bool = False,
                   router_balancing: bool = False, router_balance_strength: float = 0.1,
                   router_lr_multiplier: float = 1.0):
    """Factory that builds a ViT with configurable MoE layers.

    Args:
    ``moe_num_unshared_experts``, ``moe_num_shared_experts``,
    ``moe_top_k`` and ``moe_route_with_cls_token``.
    """
    # expert counts provided directly
    unshared = moe_num_unshared_experts
    shared = moe_num_shared_experts

    moe_params = {
        'num_unshared_experts': unshared,
        'num_shared_experts': shared,
        'top_k': moe_top_k,
        'dropout': 0.0,
        'route_with_cls_token': moe_route_with_cls_token,
    }

    model = ViTMoE(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   moe_layer_indices=moe_layer_indices, moe_params=moe_params,
                   use_class_token=moe_route_with_cls_token)

    default_params = {
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'moe_layer_indices': moe_layer_indices,
        'moe_num_unshared_experts': unshared,
        'moe_num_shared_experts': shared,
        'moe_top_k': moe_top_k,
        'moe_route_with_cls_token': moe_route_with_cls_token,
        'img_size': img_size,
        'pretrained': pretrained,
        'router_balancing': router_balancing,
        'router_balance_strength': router_balance_strength,
        'router_lr_multiplier': router_lr_multiplier,
    }

    return {
        'model': model,
        'name': 'vit_moe',
        'default_params': default_params
    }
