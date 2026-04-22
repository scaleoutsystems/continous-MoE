import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

"""
================================================================================
Hybrid MoE ViT with Multi-Timescale Learning and Prototype-Based Routing
================================================================================

OVERVIEW
--------
This model is a Vision Transformer (ViT) where every FFN layer is replaced with a
Mixture-of-Experts (MoE) layer. Routing is performed by a hybrid mechanism:

    (1) A learned projection (router network)
    (2) A set of prototype vectors (cluster centers)

Each transformer layer operates at a different *timescale*, meaning:
    - Shallow layers update slowly (stable, long-term memory)
    - Deep layers update quickly (adaptive, task-specific)

Additionally, prototypes are updated using EMA, making them behave like
slow-moving cluster centroids over time.

--------------------------------------------------------------------------------
FORWARD PASS STRUCTURE
--------------------------------------------------------------------------------

Input image
    ↓
Patch embedding + positional encoding
    ↓
[ Transformer Blocks with MoE FFN ]
    ↓
LayerNorm
    ↓
Classifier head


Each Transformer Block:

    x
    ↓
    LayerNorm
    ↓
    Multi-Head Attention
    ↓
    Residual Add
    ↓
    LayerNorm
    ↓
    Hybrid MoE (replaces FFN)
    ↓
    Residual Add


--------------------------------------------------------------------------------
MOE LAYER STRUCTURE
--------------------------------------------------------------------------------

For each token feature vector x:

        x
        ↓
   router_net (learned linear projection)
        ↓
        z (normalized)
        ↓
   cosine similarity with prototypes
        ↓
   softmax (temperature-scaled)
        ↓
   select top-1 expert (Switch routing)
        ↓
   pass x through selected expert MLP


Diagram:

              ┌──────────────┐
              │  router_net  │  (learned)
              └──────┬───────┘
                     ↓
                     z
                     ↓
        ┌────────────────────────────┐
        │ cosine(z, prototypes p_i) │
        └───────────┬───────────────┘
                    ↓
                 softmax
                    ↓
              argmax (top-1)
                    ↓
        ┌────────────┴────────────┐
        │ selected expert MLP_i   │
        └────────────┬────────────┘
                     ↓
                   output


--------------------------------------------------------------------------------
PROTOTYPES (CLUSTER CENTERS)
--------------------------------------------------------------------------------

- One prototype per expert
- Stored as normalized vectors
- Initialized from real data (warm start)
- Updated via EMA:

    p_i ← α p_i + (1 - α) mean(features assigned to expert i)

Interpretation:
    - Prototypes ≈ cluster centroids in feature space
    - Routing ≈ nearest-cluster assignment
    - Experts specialize to regions of the feature space

--------------------------------------------------------------------------------
CLUSTERING + SUBSTRUCTURE EMERGENCE
--------------------------------------------------------------------------------

The system forms a hierarchical structure:

GLOBAL FEATURE SPACE
    ↓
    Partitioned into clusters (prototypes)
        ↓
        Each cluster assigned to an expert
            ↓
            Each expert learns substructure

Visualization:

        Feature Space
        ┌──────────────────────────────┐
        │      ○ p₁        ○ p₂        │
        │        \        /            │
        │         \      /             │
        │   data → ●●●●●●●●            │
        │         /      \             │
        │        /        \            │
        │      ○ p₃        ○ p₄        │
        └──────────────────────────────┘

Each prototype:
    - defines a region (cluster)
    - routes inputs to its expert

Each expert:
    - models fine-grained structure within its region

So:
    prototypes → coarse partitioning
    experts    → fine specialization


--------------------------------------------------------------------------------
MULTI-TIMESCALE LEARNING
--------------------------------------------------------------------------------

Each layer updates at a different frequency:

    Shallow layers: update every 128 steps (slow)
    Deep layers:    update every 1 step   (fast)

Also:
    - Shallow prototypes: high EMA (e.g. 0.999) → very stable
    - Deep prototypes:    lower EMA (e.g. 0.98) → adaptive

Effect:

    Layer depth → behavior

    Shallow:
        - stable features
        - long-term memory
        - resistant to forgetting

    Deep:
        - fast adaptation
        - task-specific specialization


--------------------------------------------------------------------------------
WHY THIS HELPS CONTINUAL LEARNING
--------------------------------------------------------------------------------

1. Prototypes act as memory:
    - store structure of past data
    - move slowly via EMA

2. Routing stability:
    - similar inputs keep going to same experts

3. Expert isolation:
    - different regions → different experts
    - reduces interference

4. Multi-timescale:
    - shallow layers preserve knowledge
    - deep layers adapt to new data

Together:

    Stable structure (prototypes + shallow layers)
    +
    Plastic adaptation (deep layers + router)

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------

This model combines:

    - Vision Transformer backbone
    - Mixture-of-Experts specialization
    - Prototype-based clustering
    - Learned routing
    - Multi-timescale optimization

Result:
    A system that learns structured partitions of feature space and assigns
    specialized experts to each region, while maintaining stability over time.
"""

# -------------------------
# Hybrid MoE with EMA prototypes
# -------------------------
class HybridMoE(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=4,
        update_every=1,
        temp=0.1,
        prototype_ema=0.99,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.update_every = update_every
        self.temp = temp
        self.prototype_ema = prototype_ema

        # experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

        # learned router
        self.router = nn.Linear(dim, dim)

        # prototypes (will be warm-started)
        protos = torch.randn(num_experts, dim)
        self.register_buffer("prototypes", F.normalize(protos, dim=1))

        self._step = 0
        self._accum_steps = 0
        self._prototypes_initialized = False

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        # -------- Warm start prototypes --------
        if not self._prototypes_initialized:
            with torch.no_grad():
                feats = F.normalize(x.reshape(-1, D), dim=-1)
                if feats.size(0) >= self.num_experts:
                    self.prototypes.copy_(feats[:self.num_experts])
                else:
                    repeat = feats.repeat(self.num_experts // feats.size(0) + 1, 1)
                    self.prototypes.copy_(repeat[:self.num_experts])
                self._prototypes_initialized = True

        # -------- Routing --------
        z = F.normalize(self.router(x), dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)

        sims = torch.matmul(z, protos.t())  # (B, N, E)
        probs = F.softmax(sims / self.temp, dim=-1)
        selected = probs.argmax(dim=-1)  # (B, N)

        # -------- Experts --------
        expert_outs = torch.stack([e(x) for e in self.experts], dim=2)  # (B, N, E, D)

        selected_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
        out = expert_outs.gather(2, selected_idx).squeeze(2)

        # -------- EMA prototype update --------
        with torch.no_grad():
            feats = F.normalize(x, dim=-1)
            feats_flat = feats.reshape(-1, D)
            selected_flat = selected.reshape(-1)

            for i in range(self.num_experts):
                mask = selected_flat == i
                if mask.any():
                    mean_feat = feats_flat[mask].mean(dim=0)

                    p_old = self.prototypes[i]
                    p_new = self.prototype_ema * p_old + (1 - self.prototype_ema) * mean_feat
                    self.prototypes[i] = F.normalize(p_new, dim=0)

        return out

    def step(self, optimizer):
        self._step += 1
        self._accum_steps += 1

        if self._step % self.update_every == 0:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad /= float(self._accum_steps)

            optimizer.step()
            optimizer.zero_grad()
            self._accum_steps = 0


# -------------------------
# Transformer Block
# -------------------------
class HybridMoEBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        num_experts=4,
        update_every=1,
        prototype_ema=0.99,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.moe = HybridMoE(
            dim,
            hidden_dim,
            num_experts=num_experts,
            update_every=update_every,
            prototype_ema=prototype_ema,
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.moe(h)

        return x


# -------------------------
# Full ViT MoE
# -------------------------
class HybridMoEViT(nn.Module):
    def __init__(
        self,
        dim=384,
        depth=6,
        num_heads=6,
        num_classes=10,
        num_experts=4,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            'vit_small_patch16_128',
            pretrained=True
        )

        self.encoder.blocks = nn.ModuleList()

        # shallow → deep: slow → fast
        update_schedule = [128, 32, 8, 4, 2, 1]

        # EMA schedule: shallow more stable, deep more adaptive
        ema_schedule = [0.999, 0.995, 0.99, 0.99, 0.985, 0.98]

        self.blocks = nn.ModuleList([
            HybridMoEBlock(
                dim=dim,
                num_heads=num_heads,
                num_experts=num_experts,
                update_every=update_schedule[i],
                prototype_ema=ema_schedule[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder.patch_embed(x)

        cls_token = self.encoder.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)


# -------------------------
# Example training loop
# -------------------------
# if __name__ == "__main__":
#     model = HybridMoEViT().cuda()

#     optimizers = [
#         torch.optim.Adam(blk.moe.parameters(), lr=1e-4)
#         for blk in model.blocks
#     ]

#     criterion = nn.CrossEntropyLoss()

#     for step in range(1000):  # dummy loop
#         x = torch.randn(8, 3, 128, 128).cuda()
#         y = torch.randint(0, 10, (8,)).cuda()

#         logits = model(x)
#         loss = criterion(logits, y)

#         loss.backward()

#         for blk, opt in zip(model.blocks, optimizers):
#             blk.moe.step(opt)

#         if step % 50 == 0:
#             print(f"Step {step}, Loss {loss.item():.4f}")