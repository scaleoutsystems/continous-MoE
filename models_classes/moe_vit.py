"""
Vision Transformer with Mixture-of-Experts (MoE) FFN layers.

- Replace selected transformer MLP/FFN layers with MoE layers.
- Configurable: number of experts per MoE layer, top-k routing, shared expert,
  which layers are MoE layers (list or pattern), and ability to freeze router
  (gating) parameters after N training batches.

The factory function `create_moe_vit(...)` returns the same package dict as the
other model factories so it plugs directly into the notebook.

Notes:
- This is a straightforward, easy-to-read MoE implementation (sparse dispatch by default).
  It performs sparse dispatch (only selected experts are executed per-sample) and is
  intended for research/debugging rather than large-scale production efficiency.
- The training function supports `router_freeze_after_batches` via kwargs.
"""
from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image-level embedder that produces a single token per image.

    For the continual / MoE experiments we route on whole-image summaries so
    tokenization into many patches is unnecessary. This module returns a
    tensor shaped (B, 1, embed_dim).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        # simple 1x1 conv to map channels -> embed_dim, followed by global avg pool
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, 1, C') where C' == embed_dim
        # B = x.shape[0]
        x = self.proj(x)              # (B, embed_dim, H, W)
        x = x.mean(dim=[2, 3], keepdim=False)  # global average pool -> (B, embed_dim)
        x = x.unsqueeze(1)            # (B, 1, embed_dim)
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

    - Routing is per-image (whole-image summary).
    - Exposes gate probabilities for use by router-balancing regularizers.
    - Tracks cumulative gate statistics for utilization metrics.
    """
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=1, shared_expert=False, dropout=0.0):
        super().__init__()
        assert num_experts >= 1
        assert top_k >= 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        # default behaviour: sparse MoE with top-1 dispatch
        self.top_k = min(top_k, num_experts)
        self.shared_expert = shared_expert

        # experts: small MLPs
        self.experts = nn.ModuleList([
            MLP(dim, hidden_dim, out_dim=dim, dropout=dropout) for _ in range(num_experts)
        ])

        # gating (router) - routes based on a per-image summary vector
        self.gate = nn.Linear(dim, num_experts)

        # simple per-forward debug stats (kept detached)
        self.register_buffer('last_importance', torch.zeros(num_experts), persistent=False)
        self.register_buffer('last_load', torch.zeros(num_experts), persistent=False)

        # cumulative stats used for utilization metrics (stored on CPU)
        self.cumulative_gate_sum = torch.zeros(num_experts)
        self.cumulative_samples = 0

        # last differentiable gate probs (kept on device, not detached)
        self._last_gate_probs = None

    def get_router_parameters(self):
        return self.gate.parameters()

    def reset_cumulative_stats(self):
        self.cumulative_gate_sum = torch.zeros(self.num_experts)
        self.cumulative_samples = 0

    def get_cumulative_stats(self):
        if self.cumulative_samples == 0:
            return {'fraction': torch.zeros(self.num_experts).tolist(), 'samples': 0}
        frac = (self.cumulative_gate_sum / float(max(1, self.cumulative_samples))).tolist()
        return {'fraction': frac, 'samples': int(self.cumulative_samples)}

    def forward(self, x):
        """Sparse dispatch based on a whole-image summary (mean over tokens).

        Args:
            x: tensor (B, N, C)
        Returns:
            out: tensor (B, N, C) — combined expert outputs per-token per-sample
        """
        B, N, C = x.shape

        # Route by whole-image summary (mean pooling over tokens) so gating
        # decisions are made per-sample rather than per-token/patch.
        summary = x.mean(dim=1)  # (B, C)
        logits = self.gate(summary)  # (B, E)

        # store differentiable gate probs (used by router-balancing loss)
        gate_probs = F.softmax(logits, dim=-1)  # (B, E)
        self._last_gate_probs = gate_probs

        # accumulate CPU-side statistics for utilization reporting (keep stats on CPU)
        # ensure both operands are on CPU to avoid device-mismatch during in-place add
        self.cumulative_gate_sum = (self.cumulative_gate_sum.cpu() + gate_probs.detach().sum(dim=0).cpu())
        self.cumulative_samples += B

        E = self.num_experts
        k = min(self.top_k, E)
        if k == 0:
            return x

        # select top-k experts per sample
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)  # (B, k)
        selected_mask = torch.zeros((B, E), device=logits.device, dtype=torch.bool)
        selected_mask.scatter_(1, topk_idx, True)

        # if shared expert is requested, always include expert 0 in the selection
        if self.shared_expert:
            selected_mask[:, 0] = True

        # compute normalized weights only among selected experts
        very_neg = -1e9
        masked_logits = logits.clone()
        masked_logits[~selected_mask] = very_neg
        weights = F.softmax(masked_logits, dim=-1) * selected_mask.float()  # (B, E)

        # store simple stats (importance: sum of weights across batch; load: token-count proxy)
        importance = weights.sum(dim=0)  # (E,) sum of weights across samples
        load = selected_mask.float().sum(dim=0) * N  # (E,) approximate token load
        self.last_importance = importance.detach().cpu()
        self.last_load = load.detach().cpu()

        # Sparse execution: compute expert outputs only for selected experts per-sample.
        out = torch.zeros_like(x)
        for b in range(B):
            sel = selected_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                out[b] = x[b]
                continue
            tokens_b = x[b]  # (N, C)
            combined = torch.zeros_like(tokens_b)
            for e in sel:
                w = weights[b, e]
                expert_out = self.experts[int(e)](tokens_b)
                combined = combined + w.unsqueeze(-1) * expert_out
            out[b] = combined

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
    def __init__(self, *, img_size=224, patch_size=224, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                 moe_layer_indices: Optional[Union[List[int], str]] = None,
                 moe_params: Optional[dict] = None,
                 use_class_token=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        # single-token (full-image) embedding — no patch/tokenization
        num_patches = 1
        self.use_class_token = use_class_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # determine which layers are MoE
        if moe_layer_indices is None:
            moe_layer_indices = []
        elif isinstance(moe_layer_indices, str):
            if moe_layer_indices == 'every_other':
                moe_layer_indices = [i for i in range(depth) if (i % 2) == 1]
            elif moe_layer_indices == 'all':
                moe_layer_indices = list(range(depth))
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

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # B = x.shape[0]
        x = self.patch_embed(x)  # (B, 1, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # single-token representation (full-image)
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

    # TODO: Add in a function that slows the learning rate of the routers' parameters. Should be a configurable multiplier on the learning rate, add to config file for moe options. Also add into the example config file. Should also be able to select when to trigger this by batch, similar to freezing.

    # TODO: Add in a function that calculates the router balance loss for each router, and defines a mask for each router's loss so it only affects the given router. If routers are frozen or router_balancing is false, should return None.

    def get_moe_utilization(self):
        """Return a list of per-MoE-layer utilization statistics (fraction per expert).

        Each element is a dict: {'layer_index': i, 'fraction': [...], 'samples': n}
        """
        results = []
        for idx, m in enumerate(self.modules()):
            if isinstance(m, MoE):
                stats = m.get_cumulative_stats() if hasattr(m, 'get_cumulative_stats') else {}
                results.append({'layer_index': idx, 'fraction': stats.get('fraction', []), 'samples': stats.get('samples', 0)})
        return results

def create_moe_vit(num_classes=10,
                   img_size=224, patch_size=224,
                   embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                   moe_layer_indices: Optional[Union[List[int], str]] = 'every_other',
                   moe_num_experts: int = 4, moe_top_k: int = 1, moe_shared_expert: bool = False,
                   pretrained: bool = False,
                   router_balancing: bool = False, router_balance_strength: float = 0.1):
    """Factory that builds a ViT with configurable MoE layers.

    Returns the standard model package dict used in the notebook.
    """
    moe_params = {
        'num_experts': moe_num_experts,
        'top_k': moe_top_k,
        'shared_expert': moe_shared_expert,
        'dropout': 0.0
    }

    model = ViTMoE(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   moe_layer_indices=moe_layer_indices, moe_params=moe_params)

    default_params = {
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'moe_layer_indices': moe_layer_indices,
        'moe_num_experts': moe_num_experts,
        'moe_top_k': moe_top_k,
        'moe_shared_expert': moe_shared_expert,
        'pretrained': pretrained,
        'router_balancing': router_balancing,
        'router_balance_strength': router_balance_strength,
    }

    return {
        'model': model,
        'name': 'vit_moe',
        'default_params': default_params
    }
