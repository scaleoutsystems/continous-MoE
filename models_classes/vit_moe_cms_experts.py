import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# CMS + MoE ViT (Google-style hybrid)
# - ViT backbone
# - Noisy Top-k routing (training only)
# - Expert capacity loss (Switch Transformer style)
# - Each expert is a CMS FFN (fast + slow memory)
# =========================================================

#                         ViT-MoE-CMS
# ┌────────────────────────────────────────────────────────┐
# │ Image → Patches → Tokens                               │
# │                                                        │
# │  Transformer Block L                                   │
# │   ├─ Attention                                         │
# │   └─ MoE-CMS FFN                                       │
# │        ├─ Router (noisy top-k)                         │
# │        ├─ Expert 1 (CMS: fast + slow)                  │
# │        ├─ Expert 2 (CMS: fast + slow)                  │
# │        └─ Expert N (CMS: fast + slow)                  │
# │                                                        │
# │  Depth-wise timescale: 2^l                             │
# │  Expert-level memory: fast/slow hierarchy              │
# │  Token-level routing: sparse MoE                       │
# │  Regularization: capacity loss                         │
# └────────────────────────────────────────────────────────┘

# General structure:
# Input Image
#     │
# Patch Embedding
#     │
# [CLS] + Tokens
#     │
# ────────────────────────────────────────────
# Transformer Block 1  (fastest adaptation)
#   ├─ Self-Attention
#   └─ MoE-CMS FFN
# ────────────────────────────────────────────
# Transformer Block 2  (slower)
#   ├─ Self-Attention
#   └─ MoE-CMS FFN
# ────────────────────────────────────────────
# Transformer Block 3  (slower still)
#   ├─ Self-Attention
#   └─ MoE-CMS FFN
# ────────────────────────────────────────────
#           ...
# ────────────────────────────────────────────
# LayerNorm
#     │
# Classifier Head

# MoE-CMS structure:
#                     Token x
#                        │
#                 Noisy Router
#           (logits + Gaussian noise)
#                        │
#              Top-k selection (k=2)
#                        │
#         ┌──────────────┼──────────────┐
#         │              │              │
#    Expert 1       Expert 2       Expert 3 ...
#    (CMS FFN)      (CMS FFN)      (CMS FFN)
#         │              │              │
#         └──── weighted mixture (normalized) ────┘
#                        │
#                  FFN output
#                        │
#               + residual connection

# CMS expert structure:
#                 Input x
#                    │
#         ┌──────────┴──────────┐
#         │                     │
#    FAST PATH             SLOW PATHS
#  (always updated)     (memory systems)
#         │                     │
#  Linear → GELU → Linear   S1, S2, S3 ...
#         │                     │
#         │             (detached forward)
#         │                     │
#         └──────────┬──────────┘
#                    │
#         sigmoid gate g(x)
#                    │
#    output = g * fast + (1-g) * slow_avg

# -------------------------
# CMS Expert
# -------------------------
class CMSExpert(nn.Module):
    def __init__(self, dim, hidden_dim, num_slow=3):
        super().__init__()

        self.fast = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.slow = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_slow)
        ])

        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        fast_out = self.fast(x)

        slow_out = 0.0
        for s in self.slow:
            slow_out = slow_out + s(x).detach()
        slow_out = slow_out / len(self.slow)

        return g * fast_out + (1 - g) * slow_out


# -------------------------
# Noisy Top-k MoE Router
# -------------------------
class NoisyTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, noise_std=1.0):
        super().__init__()
        self.w = nn.Linear(dim, num_experts)
        self.noise_std = noise_std

    def forward(self, x, training=True):
        logits = self.w(x)

        if training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs


# -------------------------
# MoE + CMS FFN
# -------------------------
class MoE_CMS_FFN(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=2, num_slow=3):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = NoisyTopKRouter(dim, num_experts)

        self.experts = nn.ModuleList([
            CMSExpert(dim, hidden_dim, num_slow)
            for _ in range(num_experts)
        ])

        # load balancing auxiliary loss scale
        self.lb_scale = 0.01

    def forward(self, x, training=True):
        B, N, D = x.shape

        probs = self.router(x, training=training)

        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        # normalize top-k weights
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x)

        # track expert load for capacity loss
        expert_load = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            idx = topk_idx[..., k]          # (B, N)
            weight = topk_vals[..., k].unsqueeze(-1)

            for b in range(B):
                for n in range(N):
                    e = idx[b, n].item()
                    expert_out = self.experts[e](x[b:b+1, n:n+1, :])
                    out[b:b+1, n:n+1, :] += weight[b:b+1, n:n+1, :] * expert_out
                    expert_load[e] += 1

        # capacity loss (Switch Transformer style)
        capacity = (B * N) / self.num_experts
        capacity_loss = torch.mean(
            F.relu(expert_load - capacity)
        )

        return out, capacity_loss


# -------------------------
# Transformer Block
# -------------------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MoE_CMS_FFN(dim, dim * 4)

    def forward(self, x, training=True):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]

        h = self.norm2(x)
        ffn_out, cap_loss = self.ffn(h, training=training)

        x = x + ffn_out
        return x, cap_loss


# -------------------------
# ViT Backbone
# -------------------------
class CMS_MoE_ViT(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, num_classes=10):
        super().__init__()

        self.patch_embed = nn.Linear(768, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim, heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, training=True):
        x = self.patch_embed(x)

        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        total_cap_loss = 0.0

        for blk in self.blocks:
            x, cap_loss = blk(x, training=training)
            total_cap_loss = total_cap_loss + cap_loss

        x = self.norm(x[:, 0])
        return self.head(x), total_cap_loss


# # -------------------------
# # TRAINING LOOP
# # -------------------------
# model = CMS_MoE_ViT()
# model.train()

# ce = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# for step, (x, y) in enumerate(loader):

#     logits, cap_loss = model(x, training=True)

#     loss = ce(logits, y)

#     total_loss = loss + model.blocks[0].ffn.lb_scale * cap_loss

#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()