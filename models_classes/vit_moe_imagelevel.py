"""
Vision Transformer with image-level Top-1 Mixture-of-Experts (MoE) layers.

Routing is performed once per image (using CLS token or mean-pool) and all
tokens for that image are dispatched to the selected expert. Each MoE layer
maintains dual-timescale centroids (fast and slow) that are updated as EMA
buffers (not part of gradient computation). The layer exposes utility hooks
expected by the experiment infrastructure (auxiliary-loss computation,
usage counters, parameter grouping helpers).

This implementation is intentionally compact and focused on the project
requirements (image-level routing, centroids, losses, usage stats).
"""
from typing import Optional, List, Union, Dict

import torch
from torch import nn
import torch.nn.functional as F
try:
    import timm
except Exception:
    timm = None


def _transfer_vit_pretrained_weights_imagelevel(src, dst, moe_layer_indices=None):
    """Transfer selected weights from a timm ViT into the image-level MoE ViT.

    Focus on patch embedding, cls token, positional embeddings, and MLP -> experts
    copying so experts are initialized identical to the pretrained FFN.
    """
    if moe_layer_indices is None:
        moe_layer_indices = []
    # patch embed
    try:
        if hasattr(src, 'patch_embed') and hasattr(dst, 'patch_embed'):
            if hasattr(src.patch_embed, 'proj') and hasattr(dst.patch_embed, 'proj'):
                with torch.no_grad():
                    dst.patch_embed.proj.weight.copy_(src.patch_embed.proj.weight)
                    if getattr(src.patch_embed.proj, 'bias', None) is not None and getattr(dst.patch_embed.proj, 'bias', None) is not None:
                        dst.patch_embed.proj.bias.copy_(src.patch_embed.proj.bias)
    except Exception:
        pass

    # cls token and pos embed
    try:
        if hasattr(src, 'cls_token') and getattr(dst, 'cls_token', None) is not None:
            if dst.cls_token.shape == src.cls_token.shape:
                with torch.no_grad():
                    dst.cls_token.copy_(src.cls_token)
        if hasattr(src, 'pos_embed') and getattr(dst, 'pos_embed', None) is not None:
            if dst.pos_embed.shape == src.pos_embed.shape:
                with torch.no_grad():
                    dst.pos_embed.copy_(src.pos_embed)
    except Exception:
        pass

    # per-block copy
    try:
        n_blocks = min(len(getattr(src, 'blocks', [])), len(getattr(dst, 'blocks', [])))
        for i in range(n_blocks):
            sblk = src.blocks[i]
            dblk = dst.blocks[i]
            # copy norms
            try:
                if hasattr(sblk, 'norm1') and hasattr(dblk, 'norm1'):
                    with torch.no_grad():
                        dblk.norm1.weight.copy_(sblk.norm1.weight)
                        dblk.norm1.bias.copy_(sblk.norm1.bias)
            except Exception:
                pass
            try:
                if hasattr(sblk, 'norm2') and hasattr(dblk, 'norm2'):
                    with torch.no_grad():
                        dblk.norm2.weight.copy_(sblk.norm2.weight)
                        dblk.norm2.bias.copy_(sblk.norm2.bias)
            except Exception:
                pass

            # copy mlp weights into ImageMoE experts or plain MLP
            try:
                if hasattr(sblk, 'mlp') and hasattr(dblk, 'mlp'):
                    sfc1 = getattr(sblk.mlp, 'fc1', None)
                    sfc2 = getattr(sblk.mlp, 'fc2', None)
                    if hasattr(dblk.mlp, 'experts'):
                        for expert in dblk.mlp.experts:
                            try:
                                with torch.no_grad():
                                    if sfc1 is not None and hasattr(expert, 'fc1'):
                                        expert.fc1.weight.copy_(sfc1.weight)
                                        expert.fc1.bias.copy_(sfc1.bias)
                                    if sfc2 is not None and hasattr(expert, 'fc2'):
                                        expert.fc2.weight.copy_(sfc2.weight)
                                        expert.fc2.bias.copy_(sfc2.bias)
                            except Exception:
                                pass
                    else:
                        try:
                            with torch.no_grad():
                                if sfc1 is not None and hasattr(dblk.mlp, 'fc1'):
                                    dblk.mlp.fc1.weight.copy_(sfc1.weight)
                                    dblk.mlp.fc1.bias.copy_(sfc1.bias)
                                if sfc2 is not None and hasattr(dblk.mlp, 'fc2'):
                                    dblk.mlp.fc2.weight.copy_(sfc2.weight)
                                    dblk.mlp.fc2.bias.copy_(sfc2.bias)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass



class PatchEmbed(nn.Module):
    def __init__(self, img_size: Optional[int] = None, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size is not None:
            if img_size % patch_size != 0:
                raise ValueError("img_size not divisible by patch_size")
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size * self.grid_size
        else:
            self.grid_size = None
            self.num_patches = None
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is None or self.img_size != H:
            if H % self.patch_size != 0:
                raise ValueError("Input size not divisible by patch_size")
            self.img_size = H
            self.grid_size = H // self.patch_size
            self.num_patches = self.grid_size * self.grid_size
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
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


class ImageMoE(nn.Module):
    """Image-level Top-1 MoE layer with dual-timescale centroids.

    - Routing: compute image representation per-sample (CLS or mean), cosine
      similarity to slow centroids, choose argmax (top-1). All tokens of an
      image are forwarded to the selected expert.
    - Centroids: `c_fast` and `c_slow` are EMA buffers (no gradients).
    - Usage counters: epoch-local counts (`_epoch_usage_counts`) used for
      per-epoch statistics and balancing.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        route_with_cls_token: bool = True,
        alpha_fast: float = 0.8,
        alpha_slow: float = 0.99,
        repulsion_k: int = 2,
        repulsion_margin: float = 0.2,
        lambda_fast: float = 0.05,
        lambda_slow: float = 0.05,
        lambda_align: float = 0.5,
        lambda_cons: float = 0.1,
        lambda_lb_init: float = 1.0,
        anneal_epochs: int = 100,
        routing_temp: Optional[float] = None,
        routing_sample: bool = False,
        detach_align_steps: int = 100,
        routing_anneal_epochs: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = int(num_experts)
        self.route_with_cls_token = bool(route_with_cls_token)

        # experts (each is a standard 2-layer MLP)
        self.experts = nn.ModuleList([MLP(dim, hidden_dim, out_dim=dim) for _ in range(self.num_experts)])

        # centroids stored as buffers (EMA updates)
        c_init = F.normalize(torch.randn(self.num_experts, dim), dim=1)
        self.register_buffer('c_fast', c_init.clone())
        self.register_buffer('c_slow', c_init.clone())
        self.register_buffer('centroids_initialized', torch.tensor(0, dtype=torch.uint8))

        # EMA rates
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)

        # loss / repulsion params
        self.repulsion_k = int(repulsion_k)
        self.repulsion_margin = float(repulsion_margin)
        self.lambda_fast = float(lambda_fast)
        self.lambda_slow = float(lambda_slow)
        self.lambda_align = float(lambda_align)
        self.lambda_cons = float(lambda_cons)
        self.lambda_lb_init = float(lambda_lb_init)
        self.anneal_epochs = int(max(1, anneal_epochs))

        # epoch-local usage counters
        self.register_buffer('_epoch_usage_counts', torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer('_cumulative_usage_counts', torch.zeros(self.num_experts, dtype=torch.long))
        # small float counter to track annealing progress (incremented at epoch boundaries)
        # use a global-step counter (incremented each training forward)
        self.register_buffer('_global_step', torch.tensor(0.0))

        # routing temperature and sampling (default: deterministic argmax)
        # routing_temp stores the current (possibly annealed) temperature.
        # routing_temp_init records the initial temperature for annealing.
        self.routing_temp_init = None if routing_temp is None else float(routing_temp)
        self.routing_temp = self.routing_temp_init
        self.routing_sample = bool(routing_sample)
        # number of epochs over which to linearly anneal routing_temp_init -> 0
        self.routing_anneal_epochs = int(routing_anneal_epochs)
        # how many global steps to keep alignment z detached (support warm-up)
        self.detach_align_steps = int(detach_align_steps)

        # caches from last forward (detached where appropriate)
        self._last_z = None
        self._last_assigned = None

    def get_expert_parameters(self):
        out = []
        for idx, e in enumerate(self.experts):
            out.append((idx, list(e.parameters()), False))
        return out

    def get_router_parameters(self):
        # routing is centroid-based (no trainable router)
        return []

    def _compute_image_repr(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.route_with_cls_token:
            return x[:, 0, :]
        else: # route via feature mean w/o CLS token
            return x[:, 1:, :].mean(dim=1)

    def _maybe_initialize_centroids(self, z: torch.Tensor):
        if int(self.centroids_initialized.item()) == 1:
            return
        B = z.size(0)
        N = self.num_experts
        if B >= N:
            init = z[:N].detach().clone()
        else:
            # if batch smaller than experts, tile or sample with replacement
            reps = []
            for i in range(N):
                reps.append(z[i % B].detach().clone())
            init = torch.stack(reps, dim=0)
        init = F.normalize(init, dim=1)
        with torch.no_grad():
            self.c_fast.copy_(init)
            self.c_slow.copy_(init)
            self.centroids_initialized.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> returns (B, T, D)"""
        B, T, D = x.shape
        z = self._compute_image_repr(x)  # (B, D)
        # ensure centroids exist
        self._maybe_initialize_centroids(z)

        # normalize for cosine sim
        z_norm = F.normalize(z, dim=1)
        c_slow_norm = F.normalize(self.c_slow, dim=1)

        # similarities and assignments (optionally temperature-scaled / sampled)
        sims = torch.matmul(z_norm, c_slow_norm.t())  # (B, N)
        # use current routing temperature (may be annealed externally)
        if self.routing_temp is None or float(self.routing_temp) <= 0.0:
            assigned = sims.argmax(dim=1)
        else:
            p = F.softmax(sims / float(self.routing_temp), dim=1)
            if self.routing_sample:
                assigned = torch.multinomial(p, num_samples=1).squeeze(1)
            else:
                assigned = p.argmax(dim=1)

        self._last_z = z.detach()
        self._last_assigned = assigned.detach()

        # update epoch usage counters (image counts) for logging only
        uniq, cnts = torch.unique(assigned, return_counts=True)
        with torch.no_grad():
            self._epoch_usage_counts[uniq] += cnts.to(self._epoch_usage_counts.device)
            self._cumulative_usage_counts[uniq] += cnts.to(self._cumulative_usage_counts.device)

        # dispatch all tokens by sorting assignments to create contiguous chunks
        out = torch.zeros_like(x)
        if assigned.numel() > 0:
            idx_sorted = assigned.argsort()
            assigned_sorted = assigned[idx_sorted]
            x_sorted = x[idx_sorted]
            z_sorted = z[idx_sorted]

            # find contiguous split points where the expert id changes
            if assigned_sorted.numel() > 1:
                diffs = (assigned_sorted[1:] != assigned_sorted[:-1]).nonzero(as_tuple=False).squeeze(1) + 1
                splits = torch.cat([
                    torch.tensor([0], device=assigned.device, dtype=torch.long),
                    diffs.to(device=assigned.device, dtype=torch.long),
                    torch.tensor([assigned_sorted.size(0)], device=assigned.device, dtype=torch.long),
                ])
            else:
                splits = torch.tensor([0, assigned_sorted.size(0)], device=assigned.device, dtype=torch.long)

            # process each contiguous chunk with the corresponding expert
            for i in range(splits.size(0) - 1):
                s = int(splits[i].item())
                e = int(splits[i + 1].item())
                expert_id = int(assigned_sorted[s].item())
                x_sel = x_sorted[s:e]
                y_sel = self.experts[expert_id](x_sel)
                out[idx_sorted[s:e]] = y_sel

            # update centroids using normalized image representations (avoid magnitude bias)
            with torch.no_grad():
                for i in range(splits.size(0) - 1):
                    s = int(splits[i].item())
                    e = int(splits[i + 1].item())
                    expert_id = int(assigned_sorted[s].item())
                    z_sel = z_sorted[s:e]
                    if z_sel.numel() == 0:
                        continue
                    mu = F.normalize(z_sel, dim=1).mean(dim=0)
                    self.c_fast[expert_id] = self.alpha_fast * self.c_fast[expert_id] + (1.0 - self.alpha_fast) * mu
                    self.c_slow[expert_id] = self.alpha_slow * self.c_slow[expert_id] + (1.0 - self.alpha_slow) * mu

        # normalize centroids and advance global step when training
        with torch.no_grad():
            self.c_fast.copy_(F.normalize(self.c_fast, dim=1))
            self.c_slow.copy_(F.normalize(self.c_slow, dim=1))
            if self.training:
                try:
                    self._global_step += 1.0
                except Exception:
                    pass

        return out

    def router_balance_loss(self, strength: float = 1.0) -> torch.Tensor:
        """Simple balancing loss based on image fractions assigned to experts."""
        # Prefer batch-level counts (from last forward) for balancing; fallback to epoch stats
        device = self._epoch_usage_counts.device
        if self._last_assigned is not None:
            counts = torch.bincount(self._last_assigned.to(device), minlength=self.num_experts).float()
        else:
            counts = self._epoch_usage_counts.float()
        total = counts.sum()
        if total == 0:
            return torch.tensor(0.0, device=self.c_slow.device)
        p = counts / total
        target = torch.full_like(p, 1.0 / float(self.num_experts))
        loss = ((p - target) ** 2).mean()
        return strength * loss

    def update_routing_temperature(self, epoch: int):
        """Update the internally-stored routing temperature given the global epoch.

        The temperature is annealed linearly from `routing_temp_init` down to
        0 over `routing_anneal_epochs`. When the temperature reaches 0 the
        routing falls back to hard argmax selection.
        """
        if self.routing_temp_init is None or self.routing_anneal_epochs <= 0:
            return
        try:
            frac = max(0.0, 1.0 - float(epoch) / float(self.routing_anneal_epochs))
            self.routing_temp = float(self.routing_temp_init) * frac
        except Exception:
            # be conservative and keep current temp on error
            pass

    def router_aux_loss(self, model_cfg: Optional[Dict] = None) -> torch.Tensor:
        """Compute auxiliary centroid-based losses for this layer.

        Returns a scalar tensor on the same device as centroids.
        """
        device = self.c_slow.device
        N = self.num_experts

        # allow overriding params from provided config dict
        if model_cfg is None:
            cfg = {}
        else:
            cfg = model_cfg
        k = int(cfg.get('moe_repulsion_k', self.repulsion_k))
        margin = float(cfg.get('moe_repulsion_margin', self.repulsion_margin))
        lambda_fast = float(cfg.get('moe_lambda_fast', self.lambda_fast))
        lambda_slow = float(cfg.get('moe_lambda_slow', self.lambda_slow))
        lambda_align = float(cfg.get('moe_lambda_align', self.lambda_align))
        lambda_cons = float(cfg.get('moe_lambda_cons', self.lambda_cons))
        lambda_lb_init = float(cfg.get('moe_lambda_lb_init', self.lambda_lb_init))
        # support annealing by steps (preferred) or epochs for backward compatibility
        anneal_steps = float(cfg.get('moe_anneal_steps', cfg.get('moe_anneal_epochs', self.anneal_epochs)))

        # L_fast and L_slow: local top-k repulsion on normalized centroids
        cf = F.normalize(self.c_fast, dim=1)
        cs = F.normalize(self.c_slow, dim=1)
        S_fast = torch.matmul(cf, cf.t())
        S_slow = torch.matmul(cs, cs.t())
        # mask diagonal
        eye = torch.eye(N, device=device, dtype=torch.bool)
        S_fast_mask = S_fast.masked_fill(eye, -10.0)
        S_slow_mask = S_slow.masked_fill(eye, -10.0)
        k = min(max(1, k), N - 1)
        topk_fast = torch.topk(S_fast_mask, k=k, dim=1).values
        topk_slow = torch.topk(S_slow_mask, k=k, dim=1).values
        L_fast = torch.clamp(topk_fast - margin, min=0.0).mean() if topk_fast.numel() > 0 else torch.tensor(0.0, device=device)
        L_slow = torch.clamp(topk_slow - margin, min=0.0).mean() if topk_slow.numel() > 0 else torch.tensor(0.0, device=device)

        # L_align: 1 - cosine(z, c_slow[assigned]) averaged over assigned images
        L_align = torch.tensor(0.0, device=device)
        if self._last_z is not None and self._last_assigned is not None:
            z = self._last_z.to(device)
            # optionally keep z detached for an initial warm-up period
            try:
                step = float(self._global_step.item()) if hasattr(self, '_global_step') else 0.0
            except Exception:
                step = 0.0
            if step < float(getattr(self, 'detach_align_steps', 0)):
                z = z.detach()
            assigned = self._last_assigned.to(device)
            # for each sample compute cosine with its assigned slow centroid
            cs_sel = cs[assigned]
            z_norm = F.normalize(z, dim=1)
            cos = (z_norm * cs_sel).sum(dim=1)
            L_align = (1.0 - cos).mean()

        # L_cons: 1 - cosine(c_fast, c_slow) averaged over experts
        cos_fs = (cf * cs).sum(dim=1)
        L_cons = (1.0 - cos_fs).mean()

        # L_balance: use batch-level counts (from last forward) for loss; fallback to epoch stats
        if self._last_assigned is not None:
            counts = torch.bincount(self._last_assigned.to(device), minlength=N).float()
        else:
            counts = self._epoch_usage_counts.float()
        total = counts.sum()
        if total == 0:
            L_balance = torch.tensor(0.0, device=device)
        else:
            p = counts / total
            u = torch.full_like(p, 1.0 / float(N))
            # avoid log(0)
            eps = 1e-9
            L_balance = (p * torch.log(torch.clamp(p, min=eps) / u)).sum()

        # annealed weight for balance (1.0 -> 0.0 over anneal_steps), annealed by global step
        step = float(self._global_step.item()) if hasattr(self, '_global_step') else 0.0
        ae = float(max(1.0, anneal_steps))
        lambda_lb = lambda_lb_init * max(0.0, 1.0 - (step / ae))

        loss = (lambda_fast * L_fast) + (lambda_slow * L_slow) + (lambda_align * L_align) + (lambda_cons * L_cons) + (lambda_lb * L_balance)
        return loss

    def get_and_reset_usage_counts(self) -> List[int]:
        vals = self._epoch_usage_counts.detach().cpu().numpy().tolist()
        # reset epoch counts and increment anneal step
        self._epoch_usage_counts.zero_()
        return vals


class TransformerBlockImgMoE(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0, use_moe=True, moe_params=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        if use_moe:
            mp = moe_params or {}
            self.mlp = ImageMoE(dim=dim, hidden_dim=hidden, **mp)
        else:
            self.mlp = MLP(dim, hidden, out_dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = x_res + x_attn
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


class ViTImageMoE(nn.Module):
    def __init__(self, *, img_size: Optional[int] = None, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                 moe_layer_indices: Optional[Union[List[int], str]] = None,
                 moe_params: Optional[dict] = None,
                 use_class_token: bool = True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.use_class_token = use_class_token
        if use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter('cls_token', None)
        self.pos_embed = None
        self.pos_drop = nn.Dropout(p=0.0)

        if moe_layer_indices is None:
            moe_layer_indices = list(range(depth))
        elif isinstance(moe_layer_indices, str):
            if moe_layer_indices == 'every_other':
                moe_layer_indices = [i for i in range(depth) if (i % 2) == 1]
            elif moe_layer_indices == 'all':
                moe_layer_indices = list(range(depth))
            else:
                moe_layer_indices = list(range(depth))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_moe = i in moe_layer_indices
            block = TransformerBlockImgMoE(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_moe=use_moe, moe_params=moe_params)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _init_pos_embed(self, seq_len: int, embed_dim: int):
        device = None
        for p in self.parameters():
            device = p.device
            break
        if device is None:
            device = torch.device('cpu')
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim, device=device))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.use_class_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        seq_len = x.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != seq_len:
            self._init_pos_embed(seq_len, x.size(2))
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        out = self.head(cls)
        return out

    # auxiliary helpers expected by the training infra ---------------------------------
    def get_router_parameters(self):
        return []

    def freeze_routing(self, freeze: bool = True):
        # nothing to freeze (centroids are buffers)
        return

    def adjust_router_learning_rate(self, optimizer: torch.optim.Optimizer, multiplier: float):
        return optimizer

    def router_balance_loss(self, strength: float = 1.0) -> torch.Tensor:
        # aggregate layer-wise balance losses
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        found = False
        device = None
        for m in self.modules():
            if isinstance(m, ImageMoE):
                device = m.c_slow.device
                loss = loss.to(device)
                loss = loss + m.router_balance_loss(strength)
                found = True
        if not found:
            return torch.tensor(0.0)
        return loss

    def router_aux_loss(self, model_cfg: Optional[dict] = None) -> torch.Tensor:
        device = None
        total = torch.tensor(0.0)
        found = False
        for m in self.modules():
            if isinstance(m, ImageMoE):
                device = m.c_slow.device
                total = total.to(device)
                total = total + m.router_aux_loss(model_cfg)
                found = True
        if not found:
            return torch.tensor(0.0)
        return total

    def get_and_reset_usage_counts(self):
        # collect per-layer usage counts; return list of per-layer lists
        out = []
        for m in self.modules():
            if isinstance(m, ImageMoE):
                out.append(m.get_and_reset_usage_counts())
        return out

    def get_cumulative_usage(self):
        """Return cumulative usage per ImageMoE layer as list of lists (cpu ints)."""
        out = []
        for m in self.modules():
            if isinstance(m, ImageMoE):
                out.append(m._cumulative_usage_counts.detach().cpu().numpy().tolist())
        return out


def create_vit_moe_imagelevel(num_classes=10,
                              img_size: Optional[int] = None, patch_size=16,
                              embed_dim=256, depth=10, num_heads=4, mlp_ratio=1.0,
                              moe_layer_indices: Optional[Union[List[int], str]] = 'all',
                              num_experts: int = 4,
                              alpha_fast: float = 0.8,
                              alpha_slow: float = 0.99,
                              repulsion_k: int = 2,
                              repulsion_margin: float = 0.2,
                              lambda_fast: float = 0.05,
                              lambda_slow: float = 0.05,
                              lambda_align: float = 0.5,
                              lambda_cons: float = 0.1,
                              lambda_lb_init: float = 1.0,
                              anneal_epochs: int = 100,
                              detach_align_steps: int = 100,
                              routing_temp: Optional[float] = None,
                              routing_anneal_epochs: int = 0,
                              patch_size_default: int = 16,
                              **kwargs):
    # support pretrained ViT variants via config kwargs
    pretrained_vit = kwargs.get('pretrained_vit', None)
    pretrained_vit_tiny_path = kwargs.get('pretrained_vit_tiny_path', None)
    if pretrained_vit in ('small', 'vit_small', 'vit_small_patch16_224'):
        patch_size = 16
        img_size = 224
        depth = 12
        embed_dim = 384
        num_heads = 6
        mlp_ratio = 4.0
    if pretrained_vit in ('tiny', 'vit_tiny') or pretrained_vit_tiny_path is not None:
        patch_size = 16
        img_size = 224
        depth = 12
        embed_dim = 192
        num_heads = 3
        mlp_ratio = 4.0

    moe_params = {
        'num_experts': int(num_experts),
        'route_with_cls_token': False,
        'alpha_fast': float(alpha_fast),
        'alpha_slow': float(alpha_slow),
        'repulsion_k': int(repulsion_k),
        'repulsion_margin': float(repulsion_margin),
        'lambda_fast': float(lambda_fast),
        'lambda_slow': float(lambda_slow),
        'lambda_align': float(lambda_align),
        'lambda_cons': float(lambda_cons),
        'lambda_lb_init': float(lambda_lb_init),
        'anneal_epochs': int(anneal_epochs),
        'detach_align_steps': int(detach_align_steps),
        'routing_temp': None if routing_temp is None else float(routing_temp),
        'routing_anneal_epochs': int(routing_anneal_epochs),
    }

    model = ViTImageMoE(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        moe_layer_indices=moe_layer_indices, moe_params=moe_params, use_class_token=True)

    # If requested, attempt to initialize from pretrained ViT weights
    try:
        if pretrained_vit in ('small', 'vit_small', 'vit_small_patch16_224') and timm is not None:
            try:
                src = timm.create_model('vit_small_patch16_224', pretrained=True)
                src.eval()
                _transfer_vit_pretrained_weights_imagelevel(src, model, moe_layer_indices=moe_layer_indices)
                model.head = nn.Linear(embed_dim, num_classes)
            except Exception:
                pass
        elif (pretrained_vit in ('tiny', 'vit_tiny')) or (pretrained_vit_tiny_path is not None):
            if timm is not None:
                try:
                    src = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100,
                                            embed_dim=192, depth=12, num_heads=3, mlp_ratio=mlp_ratio)
                    if pretrained_vit_tiny_path is not None:
                        try:
                            sd = torch.load(pretrained_vit_tiny_path, map_location='cpu')
                            if isinstance(sd, dict) and ('state_dict' in sd):
                                sd = sd['state_dict']
                            try:
                                src.load_state_dict(sd, strict=False)
                            except Exception:
                                src.load_state_dict(sd)
                        except Exception:
                            pass
                    src.eval()
                    _transfer_vit_pretrained_weights_imagelevel(src, model, moe_layer_indices=moe_layer_indices)
                    model.head = nn.Linear(embed_dim, num_classes)
                except Exception:
                    pass
    except Exception:
        pass

    default_params = {
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'moe_layer_indices': moe_layer_indices,
        'num_experts': num_experts,
        'alpha_fast': alpha_fast,
        'alpha_slow': alpha_slow,
        'repulsion_k': repulsion_k,
        'repulsion_margin': repulsion_margin,
        'lambda_fast': lambda_fast,
        'lambda_slow': lambda_slow,
        'lambda_align': lambda_align,
        'lambda_cons': lambda_cons,
        'lambda_lb_init': lambda_lb_init,
        'anneal_epochs': anneal_epochs,
        'detach_align_steps': detach_align_steps,
        'routing_temp': routing_temp,
        'routing_anneal_epochs': routing_anneal_epochs,
        'img_size': img_size,
    }

    return {'model': model, 'name': 'vit_moe_imagelevel', 'default_params': default_params}
