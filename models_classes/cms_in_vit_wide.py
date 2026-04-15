import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CMS MULTI-SLOW FFN ViT (FIXED)
# =========================

class CMSMultiFFN(nn.Module):
    def __init__(self, dim, hidden_dim, num_slow=3, base_freq=2):
        super().__init__()

        self.num_slow = num_slow

        # fast learner (updated every step)
        self.fast = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # slow learners (updated via separate loss only)
        self.slow = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_slow)
        ])

        # gating over experts
        self.gate = nn.Linear(dim, num_slow + 1)

        self.freqs = [base_freq ** (i + 1) for i in range(num_slow)]

    def forward(self, x):
        fast_out = self.fast(x)

        # IMPORTANT: slow experts do NOT receive gradient from main loss
        slow_outs = [s(x).detach() for s in self.slow]

        outs = torch.stack([fast_out] + slow_outs, dim=2)  # (B, N, K, D)

        g = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)  # (B,N,K+1,1)

        out = (g * outs).sum(dim=2)
        return out


class Block(nn.Module):
    def __init__(self, dim, heads, num_slow=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CMSMultiFFN(dim, dim * 4, num_slow=num_slow)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class CMSViT(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, num_classes=10, num_slow=3):
        super().__init__()

        self.patch_embed = nn.Linear(768, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            Block(dim, heads, num_slow=num_slow)
            for _ in range(depth)
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


# # =========================
# # TRAINING SETUP
# # =========================

# model = CMSViT()
# model.train()

# criterion = nn.CrossEntropyLoss()

# # fast params (everything except slow experts)
# fast_params = [
#     p for n, p in model.named_parameters()
#     if "slow" not in n
# ]
# optimizer_fast = torch.optim.Adam(fast_params, lr=3e-4)

# # slow optimizers per frequency group
# slow_param_groups = {}
# for blk in model.blocks:
#     for k, freq in enumerate(blk.ffn.freqs):
#         slow_param_groups.setdefault(freq, []).extend(
#             list(blk.ffn.slow[k].parameters())
#         )

# slow_optimizers = {
#     f: torch.optim.Adam(params, lr=1e-4)
#     for f, params in slow_param_groups.items()
# }


# # =========================
# # TRAINING LOOP (FIXED)
# # =========================

# global_step = 0

# for x, y in loader:
#     global_step += 1

#     # ---------------------
#     # MAIN LOSS (fast model)
#     # ---------------------
#     logits = model(x)
#     loss = criterion(logits, y)

#     optimizer_fast.zero_grad()
#     loss.backward()
#     optimizer_fast.step()

#     # ---------------------
#     # SLOW CONSISTENCY LOSS
#     # (isolated graph, slow-only gradients)
#     # ---------------------
#     slow_loss = torch.tensor(0.0, device=x.device)

#     x_detached = x.detach()

#     for blk in model.blocks:
#         h = blk.norm2(blk.ffn.fast[0](x_detached))  # shared input proxy

#         fast_out = blk.ffn.fast(h).detach()

#         slow_outs = [s(h) for s in blk.ffn.slow]

#         mix = torch.stack([fast_out] + slow_outs, dim=2)

#         g = torch.softmax(blk.ffn.gate(h), dim=-1).unsqueeze(-1)

#         pred = (g * mix).sum(dim=2)

#         slow_loss = slow_loss + F.mse_loss(pred, fast_out)

#     # ---------------------
#     # SLOW OPTIMIZATION (multi-timescale)
#     # ---------------------
#     for freq, opt in slow_optimizers.items():
#         if global_step % freq == 0:
#             opt.zero_grad()
#             slow_loss.backward()
#             opt.step()