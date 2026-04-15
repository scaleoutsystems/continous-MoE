import torch
import torch.nn as nn
# One brain, multiple memory speed depths. No MoE outside of the split rates. Ensemble.
# Overall data flow:
# Input Image
#     │
# Patch Embedding
#     │
# [CLS] token concat
#     │
#  ┌───────────────────────────────┐
#  │ Transformer Block 1           │  (fast freq = 1)
#  │  ├─ Self-Attention            │
#  │  └─ CMS-FFN                   │
#  └───────────────────────────────┘
#     │
#  ┌───────────────────────────────┐
#  │ Transformer Block 2           │  (freq = 2)
#  │  ├─ Self-Attention            │
#  │  └─ CMS-FFN                   │
#  └───────────────────────────────┘
#     │
#  ┌───────────────────────────────┐
#  │ Transformer Block 3           │  (freq = 4)
#  │  ├─ Self-Attention            │
#  │  └─ CMS-FFN                   │
#  └───────────────────────────────┘
#     │
#           ...
#     │
# LayerNorm
#     │
# Classifier Head

# CMS block structure:
#                 Input x
#                    │
#         ┌──────────┴──────────┐
#         │                     │
#    Fast Path            Slow Path
#  (updated every step)  (updated rarely)
#         │                     │
#    Linear → GELU → Linear     Linear → GELU → Linear
#         │                     │
#         └──────────┬──────────┘
#                    │
#              Gating (sigmoid)
#                    │
#         output = g * fast + (1-g) * slow

# Step t:
#   ├─ Fast weights:      ALWAYS updated
#   ├─ Slow weights:
#   │     Block 1 → every 1 step
#   │     Block 2 → every 2 steps
#   │     Block 3 → every 4 steps
#   │     ...
#   └─ Loss:
#         CE (task)
#       + λ * consistency(fast vs slow)

# -----------------------------
# CMS-FFN BLOCK
# -----------------------------
class CMSFFN(nn.Module):
    def __init__(self, dim, hidden_dim, update_freq=1):
        super().__init__()

        self.fast = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.slow = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.gate = nn.Linear(dim, 1)
        self.update_freq = update_freq

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        return g * self.fast(x) + (1 - g) * self.slow(x).detach()


# -----------------------------
# TRANSFORMER BLOCK
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, heads, update_freq):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CMSFFN(dim, dim * 4, update_freq)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------
# ViT MODEL (DEPTH-WISE CMS)
# -----------------------------
class CMSViT(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, num_classes=10):
        super().__init__()

        self.patch_embed = nn.Linear(768, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim, heads, update_freq=2 ** i)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            x = blk(x)

        return self.head(self.norm(x[:, 0]))


### Training example setup:
# model = CMSViT()
# model.train()

# ce = nn.CrossEntropyLoss()

# # -----------------------
# # FAST optimizer
# # -----------------------
# fast_params = []
# for blk in model.blocks:
#     fast_params += list(blk.ffn.fast.parameters()) + list(blk.ffn.gate.parameters())

# opt_fast = torch.optim.Adam(fast_params, lr=3e-4)

# # -----------------------
# # SLOW optimizers (per block)
# # -----------------------
# opt_slow = [
#     torch.optim.Adam(blk.ffn.slow.parameters(), lr=1e-4)
#     for blk in model.blocks
# ]

# global_step = 0

# for x, y in loader:
#     global_step += 1

#     # -------------------------
#     # forward (SINGLE PASS)
#     # -------------------------
#     logits = model(x)
#     loss_main = ce(logits, y)

#     # -------------------------
#     # FAST UPDATE (every step)
#     # -------------------------
#     opt_fast.zero_grad()
#     loss_main.backward(retain_graph=True)
#     opt_fast.step()

#     # -------------------------
#     # SLOW UPDATE (multi-timescale)
#     # -------------------------
#     for i, blk in enumerate(model.blocks):
#         freq = blk.ffn.update_freq

#         if global_step % freq == 0:
#             opt_slow[i].zero_grad()

#             with torch.no_grad():
#                 slow_pred = blk.ffn.slow(blk.norm2(x))

#             fast_pred = blk.ffn.fast(blk.norm2(x))

#             slow_loss = F.mse_loss(fast_pred, slow_pred)

#             slow_loss.backward()
#             opt_slow[i].step()