import torch
import torch.nn as nn

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



# --- CMS-style FFN block ---
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

        # gating (context-conditioned mixing)
        self.gate = nn.Linear(dim, 1)

        self.update_freq = update_freq
        self.step = 0

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))  # (B, N, 1)

        fast_out = self.fast(x)
        slow_out = self.slow(x).detach()  # stop gradient unless updated

        return g * fast_out + (1 - g) * slow_out

    def slow_parameters(self):
        return self.slow.parameters()

    def fast_parameters(self):
        return list(self.fast.parameters()) + list(self.gate.parameters())

# --- Transformer block ---
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, update_freq=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = CMSFFN(dim, int(dim * mlp_ratio), update_freq)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]

        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

# --- ViT classifier ---
class CMSViT(nn.Module):
    def __init__(self, dim=256, depth=6, num_heads=8, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Linear(768, dim)  # assume flattened patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim, num_heads, update_freq=2**i)
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

# Example training loop

# model = CMSViT()
# ce_loss = nn.CrossEntropyLoss()

# # separate optimizers
# fast_params = []
# slow_param_groups = []

# for blk in model.blocks:
#     fast_params += blk.ffn.fast_parameters()
#     slow_param_groups.append({
#         "params": blk.ffn.slow_parameters(),
#         "freq": blk.ffn.update_freq,
#         "counter": 0
#     })

# optimizer_fast = torch.optim.Adam(fast_params, lr=3e-4)
# optimizer_slow = torch.optim.Adam(
#     [p for g in slow_param_groups for p in g["params"]],
#     lr=1e-4
# )

# for step, (x, y) in enumerate(loader):
#     logits = model(x)

#     # --- main task loss ---
#     loss = ce_loss(logits, y)

#     # --- auxiliary consistency loss ---
#     aux = 0.0
#     for blk in model.blocks:
#         ffn = blk.ffn
#         with torch.no_grad():
#             slow_out = ffn.slow(x)
#         fast_out = ffn.fast(x)
#         aux += F.mse_loss(fast_out, slow_out)

#     total_loss = loss + 0.1 * aux

#     # --- update fast weights every step ---
#     optimizer_fast.zero_grad()
#     total_loss.backward(retain_graph=True)
#     optimizer_fast.step()

#     # --- update slow weights at different frequencies ---
#     for g in slow_param_groups:
#         g["counter"] += 1
#         if g["counter"] % g["freq"] == 0:
#             optimizer_slow.zero_grad()
#             loss.backward()  # only main loss
#             optimizer_slow.step()


### Pseudo-code algo:
# Algorithm 1: Layer-wise Multi-Timescale CMS-Transformer

# Input: dataset D, model f with L Transformer blocks
#        update frequencies {F1, F2, ..., FL}
#        loss function L(·), optimizer θ_fast, θ_slow

# Initialize:
#     for each block l ∈ {1..L}:
#         initialize fast FFN parameters θ_f^l
#         initialize slow FFN parameters θ_s^l
#         set step counter t = 0

# for each minibatch (x, y) in D do
#     t ← t + 1

#     # Forward pass
#     h ← PatchEmbed(x)
#     for l = 1..L do
#         h ← TransformerBlock_l(h)

#     logits ← Classifier(h)
#     loss_main ← CrossEntropy(logits, y)

#     # Optional consistency loss
#     loss_aux ← Σ_l || FFN_fast^l(h) − FFN_slow^l(h) ||^2

#     loss ← loss_main + λ * loss_aux

#     # Update fast parameters (every step)
#     θ_fast ← OptimizerStep(∇_θ_fast loss)

#     # Update slow parameters at layer-dependent frequencies
#     for l = 1..L do
#         if t mod F_l == 0 then
#             θ_slow^l ← OptimizerStep(∇_θ_slow^l loss_main)
#         end if
#     end for

# end for