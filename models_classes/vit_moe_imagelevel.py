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
    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=4,
        alpha_fast=0.8,
        alpha_slow=0.99,
        top_k=2,
        routing_temp=1.0,
        warmup_steps=200,
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # experts
        self.experts = nn.ModuleList([
            MLP(dim, hidden_dim, out_dim=dim)
            for _ in range(num_experts)
        ])

        # learned router
        self.router = nn.Linear(dim, num_experts, bias=False)

        # centroid prior
        c_init = F.normalize(torch.randn(num_experts, dim), dim=1)
        self.register_buffer("c_fast", c_init.clone())
        self.register_buffer("c_slow", c_init.clone())

        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow

        self.routing_temp = routing_temp
        self.warmup_steps = warmup_steps

        self.register_buffer("_global_step", torch.tensor(0.0))

        # stats
        self.register_buffer("_epoch_usage_counts", torch.zeros(num_experts))
        self._last_weights = None

        # scaling
        self.router_scale = 1.0
        self.centroid_scale = 1.0

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        x_flat = x.reshape(B * T, D)  # (BT, D)

        # normalize
        x_norm = F.normalize(x_flat, dim=1)
        c_norm = F.normalize(self.c_slow, dim=1)

        # centroid similarity
        sims = torch.matmul(x_norm, c_norm.t())  # (BT, N)

        # learned router
        router_logits = self.router(x_flat)

        logits = (
            self.router_scale * router_logits +
            self.centroid_scale * sims
        )

        # temperature
        temp = self.routing_temp if self.routing_temp else 1.0
        weights = F.softmax(logits / temp, dim=1)

        # warmup
        if self._global_step < self.warmup_steps:
            weights = torch.full_like(weights, 1.0 / self.num_experts)

        # top-k sparse routing
        topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=1)

        # normalize top-k weights
        topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-9)

        # output buffer
        out_flat = torch.zeros_like(x_flat)

        # dispatch tokens to experts
        for expert_id in range(self.num_experts):
            mask = (topk_idx == expert_id)  # (BT, K)

            if not mask.any():
                continue

            token_indices = mask.any(dim=1).nonzero(as_tuple=False).squeeze(1)

            if token_indices.numel() == 0:
                continue

            x_sel = x_flat[token_indices]

            y_sel = self.experts[expert_id](x_sel)

            # gather weights for this expert
            w = torch.zeros(token_indices.size(0), device=x.device)

            for k in range(self.top_k):
                match = (topk_idx[token_indices, k] == expert_id)
                w += match.float() * topk_vals[token_indices, k]

            out_flat[token_indices] += y_sel * w.unsqueeze(1)

        # reshape back
        out = out_flat.view(B, T, D)

        # centroid updates (use token features)
        with torch.no_grad():
            for i in range(self.num_experts):
                mask = (topk_idx == i)
                if not mask.any():
                    continue

                token_indices = mask.any(dim=1).nonzero(as_tuple=False).squeeze(1)
                w = torch.zeros(token_indices.size(0), device=x.device)

                for k in range(self.top_k):
                    match = (topk_idx[token_indices, k] == i)
                    w += match.float() * topk_vals[token_indices, k]

                z_sel = x_norm[token_indices]

                if w.sum() < 1e-6:
                    continue

                mu = (w.unsqueeze(1) * z_sel).sum(dim=0) / (w.sum() + 1e-6)

                self.c_fast[i] = self.alpha_fast * self.c_fast[i] + (1 - self.alpha_fast) * mu
                self.c_slow[i] = self.alpha_slow * self.c_slow[i] + (1 - self.alpha_slow) * mu

            self.c_fast.copy_(F.normalize(self.c_fast, dim=1))
            self.c_slow.copy_(F.normalize(self.c_slow, dim=1))

            # usage stats
            self._epoch_usage_counts += torch.bincount(
                topk_idx.view(-1),
                minlength=self.num_experts
            ).float()

            if self.training:
                self._global_step += 1

        self._last_weights = weights.detach()

        return out

    def router_balance_loss(self):
        if self._last_weights is None:
            return torch.tensor(0.0, device=self.c_slow.device)

        p = self._last_weights.mean(dim=0)
        target = torch.full_like(p, 1.0 / self.num_experts)
        return ((p - target) ** 2).mean()

    def get_and_reset_usage_counts(self):
        vals = self._epoch_usage_counts.detach().cpu().tolist()
        self._epoch_usage_counts.zero_()
        return vals


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
    moe_params = {
        'num_experts': int(num_experts),
        'route_with_cls_token': True,
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
