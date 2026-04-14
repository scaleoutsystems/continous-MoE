import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# ViT Encoder
# -------------------------
class ViTEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=4 * dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=6)

    def forward(self, x):
        x = self.patch_embed(x)          # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        x = self.transformer(x)
        return x.mean(dim=1)             # [B, D]


# -------------------------
# Small CMS per expert (compact prototype memory)
# -------------------------
class CMS(nn.Module):
    def __init__(self, dim=256, mem_slots=32, num_heads=4):
        super().__init__()

        self.memory = nn.Parameter(torch.randn(mem_slots, dim))
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, D]
        B, D = x.shape
        M = self.memory  # [S, D]

        Q = self.q(x).view(B, self.num_heads, self.head_dim)
        K = self.k(M).view(-1, self.num_heads, self.head_dim)
        V = self.v(M).view(-1, self.num_heads, self.head_dim)

        attn = torch.einsum("bhd,shd->bhs", Q, K)
        attn = F.softmax(attn / (self.head_dim ** 0.5), dim=-1)

        context = torch.einsum("bhs,shd->bhd", attn, V)
        context = context.reshape(B, D)

        return self.out(context), attn

    @torch.no_grad()
    def update(self, x, lr=0.05):
        # winner-take-soft update (compact EMA prototypes)
        sim = torch.matmul(x, self.memory.T)  # [B, S]
        idx = sim.argmax(dim=-1)

        for b in range(x.size(0)):
            j = idx[b]
            self.memory[j] = (1 - lr) * self.memory[j] + lr * x[b]


# -------------------------
# Expert (FFN + its own CMS)
# -------------------------
class Expert(nn.Module):
    def __init__(self, dim=256, mem_slots=32):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        self.cms = CMS(dim, mem_slots=mem_slots)

    def forward(self, x):
        h = self.ffn(x)
        ctx, attn = self.cms(h)

        # residual fusion
        out = x + h + ctx
        return out, attn


# -------------------------
# MoE Router
# -------------------------
class Router(nn.Module):
    def __init__(self, dim=256, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.gate(x), dim=-1)


# -------------------------
# ViT-MoE-CMS Model
# -------------------------
class ViTMoECMS(nn.Module):
    def __init__(self, dim=256, num_experts=4, num_classes=10, mem_slots=32):
        super().__init__()

        self.encoder = ViTEncoder(dim)
        self.router = Router(dim, num_experts)

        self.experts = nn.ModuleList([
            Expert(dim, mem_slots=mem_slots) for _ in range(num_experts)
        ])

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)  # [B, D]

        gates = self.router(x)  # [B, E]

        expert_outputs = []
        all_attn = []

        for i, expert in enumerate(self.experts):
            out, attn = expert(x)
            expert_outputs.append(out)
            all_attn.append(attn)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]

        fused = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)

        logits = self.classifier(fused)

        return logits, gates, all_attn