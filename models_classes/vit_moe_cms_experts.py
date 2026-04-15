import torch
import torch.nn as nn
import torch.nn.functional as F
# Multiple CMS per layer. Each expert is a CMS FFN instead of just one CMS with multiple slow.
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
# │  Width-wise timescale: 2^l                             │
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
# Transformer Block 1  
#   ├─ Self-Attention
#   └─ MoE-CMS FFN
# ────────────────────────────────────────────
# Transformer Block 2  
#   ├─ Self-Attention
#   └─ MoE-CMS FFN
# ────────────────────────────────────────────
# Transformer Block 3  
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

        # 2 logits: [fast, slow]
        self.gate = nn.Linear(dim, 2)

    def forward(self, x):
        # (B, N, 2)
        g = torch.softmax(self.gate(x), dim=-1)

        fast_out = self.fast(x)

        slow_out = 0.0
        for s in self.slow:
            slow_out = slow_out + s(x).detach()
        slow_out = slow_out / len(self.slow)

        # split weights
        g_fast = g[..., 0].unsqueeze(-1)
        g_slow = g[..., 1].unsqueeze(-1)

        return g_fast * fast_out + g_slow * slow_out


# -------------------------
# Noisy Top-k MoE Router
# -------------------------
class NoisyRouter(nn.Module):
    def __init__(self, dim, num_experts, noise_std=1.0):
        super().__init__()
        self.linear = nn.Linear(dim, num_experts)
        self.noise_std = noise_std

    def forward(self, x):
        logits = self.linear(x)

        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        return F.softmax(logits, dim=-1)


# -------------------------
# MoE + CMS FFN
# -------------------------
class MoE_CMS_FFN(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=2, num_slow=3):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = NoisyRouter(dim, num_experts)

        self.experts = nn.ModuleList([
            CMSExpert(dim, hidden_dim, num_slow)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, N, D = x.shape

        probs = self.router(x)

        topk_val, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_val = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x)

        # capacity tracking
        load = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            idx = topk_idx[..., k]              # (B, N)
            w = topk_val[..., k].unsqueeze(-1) # (B, N, 1)

            for b in range(B):
                for n in range(N):
                    e = int(idx[b, n])  # ensure int

                    token = x[b:b+1, n:n+1, :]          # (1,1,D)
                    expert_out = self.experts[e](token) # (1,1,D)
                    expert_out = expert_out.squeeze(0).squeeze(0)  # (D)

                    out[b, n] += w[b, n] * expert_out

                    load[e] += 1

        capacity = (B * N) / self.num_experts
        capacity_loss = F.relu(load - capacity).mean()

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

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]

        h = self.norm2(x)
        ffn_out, cap_loss = self.ffn(h)

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

    def forward(self, x):
        x = self.patch_embed(x)

        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        total_cap = 0.0

        for blk in self.blocks:
            x, cap = blk(x)
            total_cap += cap

        x = self.norm(x[:, 0])
        return self.head(x), total_cap
    

### Example training setup:
# model = CMS_MoE_ViT()
# model.train()

# ce = nn.CrossEntropyLoss()

# # -------------------------
# # 1. FAST OPTIMIZER (WITH ROUTER LR SCALING)
# # -------------------------
# router_params = []
# fast_params = []

# for blk in model.blocks:
#     # separate router params
#     router_params += list(blk.ffn.router.parameters())

#     for expert in blk.ffn.experts:
#         fast_params += list(expert.fast.parameters())
#         fast_params += list(expert.gate.parameters())

# opt_fast = torch.optim.Adam([
#     {"params": fast_params, "lr": 3e-4},
#     {"params": router_params, "lr": 3e-5},  # 0.1× LR
# ])

# # -------------------------
# # 2. SLOW OPTIMIZERS (PER EXPERT GROUP)
# # -------------------------
# slow_opts = []

# for blk in model.blocks:
#     for expert in blk.ffn.experts:
#         slow_opts.append(
#             torch.optim.Adam(expert.slow.parameters(), lr=1e-4)
#         )

# # -------------------------
# # TRAINING LOOP
# # -------------------------
# global_step = 0
# lambda_cap = 0.01

# for x, y in loader:
#     global_step += 1

#     # ---------------------
#     # forward
#     # ---------------------
#     logits, cap_loss = model(x)

#     loss = ce(logits, y)
#     total_loss = loss + lambda_cap * cap_loss

#     # ---------------------
#     # FAST UPDATE
#     # ---------------------
#     opt_fast.zero_grad()
#     total_loss.backward(retain_graph=True)
#     opt_fast.step()

#     # ---------------------
#     # SLOW UPDATE (MULTI-SCALE)
#     # ---------------------
#     for i, opt in enumerate(slow_opts):

#         freq = 2 ** ((i % 3) + 1) # shared CMS schedule per expert index

#         if global_step % freq == 0:

#             opt.zero_grad()

#             # recompute forward for stable slow gradient
#             logits, cap_loss = model(x)

#             slow_loss = ce(logits, y) + lambda_cap * cap_loss

#             slow_loss.backward()
#             opt.step()