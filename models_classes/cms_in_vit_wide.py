import torch
import torch.nn as nn

    #              CMS Multi-FFN

    #                   Input
    #                     │
    #     ┌───────────────┼────────────────┼────────────────┐
    #     │               │                │
    #  Fast (t=1)    Slow (t=2^1)   Slow (t=2^2)   Slow (t=2^3)
    #     │               │                │                │
    #     └───────┬───────┴───────┬────────┴───────┬────────┘
    #             │               │                │
    #          Gating (softmax over all learners)
    #                     │
    #                 Output mix

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CMS MULTI-SLOW FFN ViT
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

        # slow learners (multi-timescale memory)
        self.slow = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_slow)
        ])

        # gating over fast + slow experts
        self.gate = nn.Linear(dim, num_slow + 1)

        # multiplicative frequency schedule: 2^1, 2^2, ...
        self.freqs = [base_freq ** (i + 1) for i in range(num_slow)]

    def forward(self, x):
        fast_out = self.fast(x)

        slow_outs = [s(x).detach() for s in self.slow]

        outs = torch.stack([fast_out] + slow_outs, dim=0)  # (K+1,B,N,D)

        g = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)

        out = (g * outs.permute(1, 2, 0, 3)).sum(dim=2)
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

# # fast optimizer (all non-slow params)
# fast_params = [
#     p for n, p in model.named_parameters()
#     if "slow" not in n
# ]
# optimizer_fast = torch.optim.Adam(fast_params, lr=3e-4)

# # slow optimizers grouped by frequency
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
# # TRAINING LOOP
# # =========================

# global_step = 0

# for x, y in loader:
#     global_step += 1

#     # ---------------------
#     # forward + main loss
#     # ---------------------
#     logits = model(x)
#     loss = criterion(logits, y)

#     # ---------------------
#     # fast update (every step)
#     # ---------------------
#     optimizer_fast.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizer_fast.step()

#     # ---------------------
#     # slow consistency loss
#     # ---------------------
#     slow_loss = 0.0

#     for blk in model.blocks:
#         f = blk.ffn.fast(blk.ffn.fast[0].weight.new_tensor(x))

#         slow_outs = [s(blk.ffn.fast[0].weight.new_tensor(x)).detach()
#                      for s in blk.ffn.slow]

#         mix = torch.stack([f] + slow_outs, dim=0)

#         g = torch.softmax(blk.ffn.gate(
#             blk.ffn.fast[0].weight.new_tensor(x)
#         ), dim=-1).unsqueeze(-1)

#         slow_loss += F.mse_loss(
#             (g * mix.permute(1, 2, 0, 3)).sum(dim=2),
#             f
#         )

#     # ---------------------
#     # slow updates (multi-timescale)
#     # ---------------------
#     for f, opt in slow_optimizers.items():
#         if global_step % f == 0:
#             opt.zero_grad()
#             slow_loss.backward()
#             opt.step()