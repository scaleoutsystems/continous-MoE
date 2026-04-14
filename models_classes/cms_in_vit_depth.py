import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, dim, hidden_dim, update_freq=8):
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

        fast_out = self.fast(x)
        slow_out = self.slow(x).detach()

        return g * fast_out + (1 - g) * slow_out

    def fast_parameters(self):
        return list(self.fast.parameters()) + list(self.gate.parameters())

    def slow_parameters(self):
        return list(self.slow.parameters())


# -----------------------------
# TRANSFORMER BLOCK
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, update_freq=8):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CMSFFN(dim, dim * 4, update_freq)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]

        h = self.norm2(x)
        x = x + self.ffn(h)

        return x


# -----------------------------
# ViT MODEL (DEPTH-WISE CMS)
# -----------------------------
class CMSViT(nn.Module):
    def __init__(self, dim=256, depth=6, num_heads=8, num_classes=10):
        super().__init__()

        self.patch_embed = nn.Linear(768, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim, num_heads, update_freq=2 ** i)
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

        x = self.norm(x[:, 0])
        return self.head(x)


# # -----------------------------
# # OPTIMIZER SETUP
# # -----------------------------
# model = CMSViT()
# ce_loss = nn.CrossEntropyLoss()

# # FAST optimizer (all fast + gate params)
# fast_params = []
# for blk in model.blocks:
#     fast_params += blk.ffn.fast_parameters()

# optimizer_fast = torch.optim.Adam(fast_params, lr=3e-4)

# # SLOW optimizers (ONE PER BLOCK = correct CMS separation)
# slow_optimizers = []
# for blk in model.blocks:
#     opt = torch.optim.Adam(blk.ffn.slow_parameters(), lr=1e-4)
#     slow_optimizers.append(opt)

# # counters for update scheduling
# step_counter = 0


# # -----------------------------
# # TRAINING LOOP
# # -----------------------------
# for step, (x, y) in enumerate(loader):
#     step_counter += 1

#     # ---------------------
#     # forward + main loss
#     # ---------------------
#     logits = model(x)
#     loss_main = ce_loss(logits, y)

#     # ---------------------
#     # auxiliary consistency loss
#     # ---------------------
#     aux = 0.0
#     for blk in model.blocks:
#         fast_out = blk.ffn.fast(x)
#         with torch.no_grad():
#             slow_out = blk.ffn.slow(x)

#         aux += F.mse_loss(fast_out, slow_out)

#     total_loss = loss_main + 0.1 * aux

#     # ---------------------
#     # FAST UPDATE (every step)
#     # ---------------------
#     optimizer_fast.zero_grad()
#     total_loss.backward(retain_graph=True)
#     optimizer_fast.step()

#     # ---------------------
#     # SLOW UPDATES (depth-wise frequencies)
#     # ---------------------
#     for i, blk in enumerate(model.blocks):
#         freq = blk.ffn.update_freq

#         if step_counter % freq == 0:
#             slow_optimizers[i].zero_grad()

#             # IMPORTANT: recompute loss graph for slow update
#             logits = model(x)
#             slow_loss = ce_loss(logits, y)

#             slow_loss.backward()
#             slow_optimizers[i].step()