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
from .replay import ReplayBuffer


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
        B = x.shape[0]
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
        self.use_class_token = False
        self.cls_token = None
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
        B = x.shape[0]
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


# Training loop for ViT-MoE (adapted from the shared continual trainer)
def train_moe(dataloader, model, loss_fn, optimizer, test_dataloader=None, test_fn=None,
              test_interval='class', test_every_n=100, class_order=None,
              router_freeze_after_batches: Optional[int] = None,
              router_balancing: bool = False, router_balance_strength: float = 0.1,
              dataset_manager=None, replay_batch_size: int = 0, replay_weight: float = 1.0):
    """Continual-stream training loop with optional router-freezing, router-balancing
    and optional replay-buffer sampling.

    This trainer now detects a shuffled DataLoader and will skip class-boundary
    testing/prints while still accumulating per-class metrics. When
    `test_interval=='class'` with shuffled training we run a single final
    evaluation so callers still get a populated `test_history`.
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

    # Detect shuffled DataLoader (RandomSampler / SubsetRandomSampler)
    is_shuffled = False
    try:
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None:
            sname = sampler.__class__.__name__
            if sname in ('RandomSampler', 'SubsetRandomSampler'):
                is_shuffled = True
    except Exception:
        is_shuffled = False

    if is_shuffled:
        print('Shuffled dataloader detected — treating training as standard shuffled training (no per-class boundaries)')

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_class = int(y[0].item())

        # freeze router when requested
        if router_freeze_after_batches is not None and batch_count == router_freeze_after_batches:
            if hasattr(model, 'freeze_routing'):
                model.freeze_routing(True)

        # Only perform class-boundary logic for class-ordered streams
        if not is_shuffled:
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
                        # reset MoE cumulative stats before testing so metrics reflect test run
                        try:
                            for m in model.modules():
                                if hasattr(m, 'reset_cumulative_stats'):
                                    m.reset_cumulative_stats()
                        except Exception:
                            pass
                        test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                        test_result['step'] = current_class
                        test_result['step_type'] = 'class'
                        test_history.append(test_result)
                        model.train()

                current_class = y_class
                print(f"Starting training on Class {current_class}")
                class_change_steps.append(batch)
        else:
            # shuffled: aggregate stats only
            pass

        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # optional router-balancing auxiliary loss (differentiable via gate probs)
        if router_balancing:
            bal_loss = 0.0
            for m in model.modules():
                if isinstance(m, MoE) and getattr(m, '_last_gate_probs', None) is not None:
                    p_mean = m._last_gate_probs.mean(dim=0)  # (E,)
                    target = torch.full_like(p_mean, 1.0 / float(max(1, m.num_experts)))
                    bal_loss = bal_loss + ((p_mean - target) ** 2).mean()
            if isinstance(bal_loss, torch.Tensor):
                loss = loss + router_balance_strength * bal_loss

        # optional replay sampling
        if hasattr(model, 'replay_buffer') and replay_batch_size > 0 and model.replay_buffer.size() > 0:
            Xr_cpu, yr_cpu = model.replay_buffer.sample(replay_batch_size)
            if Xr_cpu.numel() > 0:
                Xr = Xr_cpu.to(device)
                yr = yr_cpu.to(device).long()
                pred_r = model(Xr)
                loss_r = loss_fn(pred_r, yr)
                loss = loss + replay_weight * loss_r

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # optionally let the dataset manager add samples to replay buffer
        if dataset_manager is not None and hasattr(dataset_manager, 'add_to_replay_if_present'):
            try:
                dataset_manager.add_to_replay_if_present(model, X.detach().cpu(), y.detach().cpu(), losses=None)
            except Exception:
                pass

        # accumulate per-class stats (works for both continual and shuffled streams)
        class_batch_counts[current_class] = class_batch_counts.get(current_class, 0) + 1
        class_losses[current_class] = class_losses.get(current_class, 0.0) + loss.item()
        batch_count += 1

        if batch % 100 == 0 and batch > 0:
            print(f"  Batch {batch}: loss: {loss.item():>7f}")
            if test_interval == 'batch' and batch % test_every_n == 0 and test_dataloader is not None and test_fn is not None:
                print(f"  Testing at Batch {batch}:")
                try:
                    for m in model.modules():
                        if hasattr(m, 'reset_cumulative_stats'):
                            m.reset_cumulative_stats()
                except Exception:
                    pass
                test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                test_result['step'] = batch
                test_result['step_type'] = 'batch'
                test_history.append(test_result)
                model.train()

    # Finalize metrics for all seen classes
    for cls in sorted(class_batch_counts.keys()):
        cnt = class_batch_counts[cls]
        avg = class_losses[cls] / cnt if cnt > 0 else float('nan')
        training_metrics[cls] = {'samples': cnt, 'avg_loss': avg}

    # If shuffled training was used and the user requested class-interval testing,
    # provide a single final evaluation (intermediate per-class tests are skipped).
    if is_shuffled:
        if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
            print('Shuffled training detected — running final evaluation (intermediate per-class tests skipped).')
            try:
                for m in model.modules():
                    if hasattr(m, 'reset_cumulative_stats'):
                        m.reset_cumulative_stats()
            except Exception:
                pass
            test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
            test_result['step'] = batch_count
            test_result['step_type'] = 'shuffled'
            test_history.append(test_result)
    else:
        # preserve original behavior for class-ordered streams
        if current_class is not None:
            avg_loss = class_losses[current_class] / class_batch_counts[current_class]
            print(f"  Class {current_class} - Processed {class_batch_counts[current_class]} samples, Avg loss: {avg_loss:>7f}")

            if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
                print(f"  Testing after Class {current_class}:")
                try:
                    for m in model.modules():
                        if hasattr(m, 'reset_cumulative_stats'):
                            m.reset_cumulative_stats()
                except Exception:
                    pass
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
                   lr: float = 1e-3, pretrained: bool = False,
                   router_balancing: bool = False, router_balance_strength: float = 0.1,
                   replay_capacity: int = 0, replay_policy: str = 'fifo'):
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

    # optionally attach a replay buffer to the model for online replay
    if replay_capacity and replay_capacity > 0:
        try:
            model.replay_buffer = ReplayBuffer(int(replay_capacity), policy=replay_policy)
        except Exception:
            model.replay_buffer = None

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
        'pretrained': pretrained,
        'router_balancing': router_balancing,
        'router_balance_strength': router_balance_strength,
        'replay_capacity': replay_capacity,
        'replay_policy': replay_policy
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
