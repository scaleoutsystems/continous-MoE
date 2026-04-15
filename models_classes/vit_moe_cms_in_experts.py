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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        return x.mean(dim=1)  # [B, D]


# -------------------------
# CMS (per expert, routing-consistent)
# -------------------------
class CMS(nn.Module):
    def __init__(self, dim=256, mem_slots=32):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        M = self.memory

        Q = self.q(x)
        K = self.k(M)
        V = self.v(M)

        attn = torch.matmul(Q, K.T) / (D ** 0.5)
        attn = F.softmax(attn, dim=-1)

        context = torch.matmul(attn, V)
        return self.out(context), attn

    @torch.no_grad()
    def update(self, x, lr=0.05):
        # routing-consistent soft prototype update
        sim = torch.matmul(x, self.memory.T)          # [B, S]
        weights = F.softmax(sim, dim=-1)              # soft assignment

        # slot-wise EMA update
        for i in range(self.memory.size(0)):
            w = weights[:, i].unsqueeze(-1)
            if w.sum() > 0:
                self.memory[i] = (
                    (1 - lr) * self.memory[i]
                    + lr * (w * x).sum(dim=0)
                )


# -------------------------
# Expert (FFN + CMS)
# -------------------------
class Expert(nn.Module):
    def __init__(self, dim=256, mem_slots=32):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.cms = CMS(dim, mem_slots)

    def forward(self, x):
        h = self.ffn(x)
        ctx, attn = self.cms(h)
        return x + h + ctx, attn

    @torch.no_grad()
    def update_cms(self, x):
        # ensure CMS sees same representation used in forward path
        h = self.ffn(x)
        self.cms.update(h)


# -------------------------
# Router (Top-K logits)
# -------------------------
class Router(nn.Module):
    def __init__(self, dim=256, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        return self.gate(x)  # logits


# -------------------------
# ViT-MoE-CMS (Top-K routing)
# -------------------------
class ViTMoECMS(nn.Module):
    def __init__(self, dim=256, num_experts=4, num_classes=10, mem_slots=32, top_k=2):
        super().__init__()
        self.encoder = ViTEncoder(dim)
        self.router = Router(dim, num_experts)

        self.experts = nn.ModuleList([
            Expert(dim, mem_slots) for _ in range(num_experts)
        ])

        self.classifier = nn.Linear(dim, num_classes)
        self.top_k = top_k

    def forward(self, x):
        x = self.encoder(x)  # [B, D]

        router_logits = self.router(x)  # [B, E]

        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        B, D = x.shape
        out = torch.zeros_like(x)

        for k in range(self.top_k):
            idx = topk_idx[:, k]                 # [B]
            weight = gates[:, k].unsqueeze(-1)   # [B, 1]

            out_k = torch.zeros_like(x)

            for e, expert in enumerate(self.experts):
                mask = (idx == e)
                if mask.any():
                    xe = x[mask]
                    ye, _ = expert(xe)
                    out_k[mask] = ye

            out += weight * out_k

        logits = self.classifier(out)

        return logits, router_logits, topk_idx


# -------------------------
# TRAINING STEP (with full routing-consistent CMS updates)
# -------------------------
def train_step(model, optimizer, x, y, lambda_entropy=0.01):
    model.train()

    logits, router_logits, topk_idx = model(x)

    # task loss
    loss_task = F.cross_entropy(logits, y)

    # router entropy (encourages exploration early)
    p = F.softmax(router_logits, dim=-1)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()

    loss = loss_task - lambda_entropy * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -------------------------------------------------
    # ROUTING-CONSISTENT CMS UPDATES (FINAL CORRECT FORM)
    # -------------------------------------------------
    with torch.no_grad():
        x_embed = model.encoder(x)

        # IMPORTANT:
        # update ONLY the CMS of experts that actually received routed samples
        for k in range(model.top_k):
            idx = topk_idx[:, k]

            for e, expert in enumerate(model.experts):
                mask = (idx == e)

                if mask.any():
                    xe = x_embed[mask]
                    expert.update_cms(xe)

    return {
        "loss": loss.item(),
        "task": loss_task.item(),
        "entropy": entropy.item()
    }