import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MultiTimescaleFFN(nn.Module):
    def __init__(self, dim, hidden_dim, update_every=1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.update_every = update_every
        self._step = 0

        # buffer for gradient scaling
        self._accum_steps = 0

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def step(self, optimizer):
        """Call after loss.backward()"""
        self._step += 1
        self._accum_steps += 1

        if self._step % self.update_every == 0:
            # normalize gradients
            for p in self.parameters():
                if p.grad is not None:
                    p.grad /= float(self._accum_steps)

            optimizer.step()
            optimizer.zero_grad()
            self._accum_steps = 0


class MoEViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, update_every=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.ffn = MultiTimescaleFFN(dim, hidden_dim, update_every)

    def forward(self, x):
        # attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # FFN (multi-timescale)
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x


class MultiTimescaleViT(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        dim=384,
        depth=6,
        num_heads=6,
        num_classes=10,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            'vit_small_patch16_128',
            pretrained=True
        )

        # remove original blocks
        self.encoder.blocks = nn.ModuleList()

        # create custom blocks with different update rates
        update_schedule = [128, 32, 8, 4, 2, 1]

        self.blocks = nn.ModuleList([
            MoEViTBlock(
                dim=dim,
                num_heads=num_heads,
                update_every=update_schedule[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)

########## Training example ##########
# model = MultiTimescaleViT().cuda()

# # separate optimizers per FFN
# optimizers = []
# for blk in model.blocks:
#     opt = torch.optim.Adam(blk.ffn.parameters(), lr=1e-4)
#     optimizers.append(opt)

# criterion = nn.CrossEntropyLoss()

# for step, (x, y) in enumerate(loader):
#     x, y = x.cuda(), y.cuda()

#     logits = model(x)
#     loss = criterion(logits, y)

#     loss.backward()

#     # step each FFN with its own schedule
#     for blk, opt in zip(model.blocks, optimizers):
#         blk.ffn.step(opt)