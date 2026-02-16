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
import math

import torch
from torch import nn
import torch.nn.functional as F

# reuse test/backward from convnext for consistency
from .convnext import test as _shared_test, backward_fn as _shared_backward


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=224, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, C)
        x = self.proj(x)  # (B, embed, H/patch, W/patch)
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


class MoE(nn.Module):
    """Sparse Mixture-of-Experts layer (sparse dispatch by default).

    - Routing is done per-sample (whole-image summary) instead of per-patch.
    - Only the selected experts are executed (sparse dispatch). This reduces
      compute compared with running every expert for every token when top-k is small.
    - Supports top-k selection per image, an optional always-included shared expert,
      and exposes router parameter accessors.
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

        # statistics for debugging / optional auxiliary losses
        self.register_buffer('last_importance', torch.zeros(num_experts), persistent=False)
        self.register_buffer('last_load', torch.zeros(num_experts), persistent=False)

    def get_router_parameters(self):
        return self.gate.parameters()

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
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.use_class_token = use_class_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_class_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if use_class_token else 0), embed_dim))
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
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)
        if self.use_class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_class_token:
            cls = x[:, 0]
        else:
            cls = x.mean(dim=1)
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


# Training loop for ViT-MoE (adapted from the shared continual trainer)
def train_moe(dataloader, model, loss_fn, optimizer, test_dataloader=None, test_fn=None,
              test_interval='class', test_every_n=100, class_order=None, router_freeze_after_batches: Optional[int] = None):
    """Continual-stream training loop with optional router-freezing support.

    Arguments follow the notebook's train(...) signature; router_freeze_after_batches
    can be used to stop updating router parameters after N batches.
    """
    model.train()
    device = next(model.parameters()).device
    batch_count = 0
    current_class = None
    class_batch_counts = {}
    class_losses = {}
    training_metrics = {}
    test_history = []
    class_change_steps = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_class = int(y[0].item())

        # freeze router when requested
        if router_freeze_after_batches is not None and batch_count == router_freeze_after_batches:
            if hasattr(model, 'freeze_routing'):
                model.freeze_routing(True)

        if current_class != y_class:
            if current_class is not None:
                avg_loss = class_losses[current_class] / class_batch_counts[current_class]
                training_metrics[current_class] = {
                    'samples': class_batch_counts[current_class],
                    'avg_loss': avg_loss
                }
                print(f"  Class {current_class} - Processed {class_batch_counts[current_class]} samples, Avg loss: {avg_loss:>7f}")

                if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
                    print(f"  Testing after Class {current_class}:")
                    test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                    test_result['step'] = current_class
                    test_result['step_type'] = 'class'
                    test_history.append(test_result)
                    model.train()

            current_class = y_class
            print(f"Starting training on Class {current_class}")
            class_change_steps.append(batch)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        class_batch_counts[current_class] = class_batch_counts.get(current_class, 0) + 1
        class_losses[current_class] = class_losses.get(current_class, 0.0) + loss.item()
        batch_count += 1

        if batch % 100 == 0 and batch > 0:
            print(f"  Batch {batch}: loss: {loss.item():>7f}")
            if test_interval == 'batch' and batch % test_every_n == 0 and test_dataloader is not None and test_fn is not None:
                print(f"  Testing at Batch {batch}:")
                test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                test_result['step'] = batch
                test_result['step_type'] = 'batch'
                test_result['current_class'] = current_class
                test_history.append(test_result)
                model.train()

    if current_class is not None:
        avg_loss = class_losses[current_class] / class_batch_counts[current_class]
        training_metrics[current_class] = {
            'samples': class_batch_counts[current_class],
            'avg_loss': avg_loss
        }
        print(f"  Class {current_class} - Processed {class_batch_counts[current_class]} samples, Avg loss: {avg_loss:>7f}")

        if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
            print(f"  Testing after Class {current_class}:")
            test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
            test_result['step'] = current_class
            test_result['step_type'] = 'class'
            test_history.append(test_result)
            model.train()

    print(f"Training complete. Total batches: {batch_count}\n")
    return training_metrics, test_history, class_change_steps


def create_moe_vit(num_classes=10, device=None,
                   img_size=224, patch_size=224,
                   embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                   moe_layer_indices: Optional[Union[List[int], str]] = 'every_other',
                   moe_num_experts: int = 4, moe_top_k: int = 1, moe_shared_expert: bool = False,
                   lr: float = 1e-3, pretrained: bool = False):
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
    if device is not None:
        model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    default_params = {
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'moe_layer_indices': moe_layer_indices,
        'moe_num_experts': moe_num_experts,
        'moe_top_k': moe_top_k,
        'moe_shared_expert': moe_shared_expert,
        'lr': lr,
        'pretrained': pretrained
    }

    return {
        'model': model,
        'loss_fn': loss,
        'optimizer': optimizer,
        'train_fn': train_moe,
        'test_fn': _shared_test,
        'backward_fn': _shared_backward,
        'name': 'vit_moe',
        'default_params': default_params
    }
