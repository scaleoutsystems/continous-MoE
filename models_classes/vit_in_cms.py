import torch
import torch.nn as nn
import torch.nn.functional as F

# Structure:
    # IMAGE
    #   ↓
    # VISION ENCODER (ViT-style)
    #   ↓
    # FEATURE REPRESENTATION (tokens / embeddings)
    #   ↓
    # CMS (memory retrieval)
    #   ↓
    # FAST ADAPTATION LAYER
    #   ↓
    # CLASSIFIER HEAD

# Training loop:
    # OUTER LOOP (controls memory + learning rules)
    #    ↓
    # MIDDLE LOOP (controls representation + routing)
    #    ↓
    # INNER LOOP (fast adaptation during inference/training)

# Inner loop structure:
    #                 IMAGE
    #                   │
    #                   ▼
    #            ┌─────────────┐
    #            │  ENCODER    │
    #            └──────┬──────┘
    #                   ▼
    #          FEATURE REPRESENTATION x
    #                   │
    #         ┌─────────┴──────────┐
    #         ▼                    ▼
    #    CMS READ              FAST LEARNER
    #    (retrieve context)    (adapt features)
    #         │                    │
    #         └─────────┬──────────┘
    #                   ▼
    #         CONTEXTUALIZED DECISION
    #         (classification logits)

# CMS Structure:
    #              ┌─────────────────────────────┐
    #              │      QUERY INPUT            │
    #              │   image feature vector x    │
    #              └─────────────┬───────────────┘
    #                            ▼
    #             ┌─────────────────────────────┐
    #             │  MEMORY ROUTER / INDEXER    │
    #             │                             │
    #             │  - similarity search        │
    #             │  - gating (what to recall)  │
    #             └─────────────┬───────────────┘
    #                            ▼
    #  ┌──────────────────────────────────────────────┐
    #  │            MEMORY SPACE (CMS)                │
    #  │                                              │
    #  │  ┌──────────────┐   ┌──────────────┐        │
    #  │  │ Prototype A   │   │ Prototype B   │  ...  │
    #  │  │ (cat-like)    │   │ (car-like)    │        │
    #  │  └──────────────┘   └──────────────┘        │
    #  │                                              │
    #  │  each = compressed evolving embedding        │
    #  └─────────────────────┬────────────────────────┘
    #                        ▼
    #           ┌─────────────────────────────┐
    #           │ MEMORY READOUT MODULE       │
    #           │                             │
    #           │ weighted aggregation:       │
    #           │   Σ attention(x, Mi)*Mi     │
    #           └─────────────┬───────────────┘
    #                         ▼
    #              CONTEXT VECTOR c(x)
# CMS Update flow:
    # image → encoder → feature x
    #                     │
    #                     ▼
    #         ┌──────────────────────────┐
    #         │ MEMORY MATCHING           │
    #         │ find nearest prototypes   │
    #         └───────────┬──────────────┘
    #                     ▼
    #         ┌──────────────────────────┐
    #         │ MEMORY UPDATE RULE        │
    #         │                          │
    #         │ M ← (1-α)M + αx          │
    #         │ or clustering merge      │
    #         └───────────┬──────────────┘
    #                     ▼
    #         ┌──────────────────────────┐
    #         │ CONSOLIDATION STEP        │
    #         │ - merge similar slots     │
    #         │ - prune outdated ones     │
    #         └──────────────────────────┘

# Fast Adaptation structure:
    #          ┌─────────────────────────────┐
    #          │   INPUT IMAGE FEATURES      │
    #          │         x (from encoder)    │
    #          └─────────────┬───────────────┘
    #                        ▼
    #          ┌─────────────────────────────┐
    #          │ CONTEXT INJECTION           │
    #          │ from CMS: c(x)             │
    #          └─────────────┬───────────────┘
    #                        ▼
    # ┌────────────────────────────────────────┐
    # │ FAST ADAPTER NETWORK                   │
    # │                                        │
    # │  x → layer norm → fusion(x, c(x))     │
    # │     → small MLP / attention block      │
    # │     → task-specific transform          │
    # └─────────────┬──────────────────────────┘
    #               ▼
    # ┌────────────────────────────────────────┐
    # │ TASK HEAD / CLASSIFIER                 │
    # │                                        │
    # │ logits = W · h(x, c(x))               │
    # └────────────────────────────────────────┘

# Update Flow:
    #          ┌─────────────────────────────┐
    #          │   INPUT IMAGE FEATURES      │
    #          │         x (from encoder)    │
    #          └─────────────┬───────────────┘
    #                        ▼
    #          ┌─────────────────────────────┐
    #          │ CONTEXT INJECTION           │
    #          │ from CMS: c(x)             │
    #          └─────────────┬───────────────┘
    #                        ▼
    # ┌────────────────────────────────────────┐
    # │ FAST ADAPTER NETWORK                   │
    # │                                        │
    # │  x → layer norm → fusion(x, c(x))     │
    # │     → small MLP / attention block      │
    # │     → task-specific transform          │
    # └─────────────┬──────────────────────────┘
    #               ▼
    # ┌────────────────────────────────────────┐
    # │ TASK HEAD / CLASSIFIER                 │
    # │                                        │
    # │ logits = W · h(x, c(x))               │
    # └────────────────────────────────────────┘

class ViTEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=4 * dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)          # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        x = self.transformer(x)
        return x.mean(dim=1)             # global feature [B, D]
    
class CMS(nn.Module):
    def __init__(self, dim=256, memory_size=1024):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, dim))
        self.memory_size = memory_size

    def forward(self, x):
        # x: [B, D]
        sim = torch.matmul(x, self.memory.T)  # [B, M]
        attn = F.softmax(sim, dim=-1)
        context = torch.matmul(attn, self.memory)  # [B, D]
        return context, attn

    @torch.no_grad()
    def update(self, x, lr=0.05):
        # simple competitive update (winner-takes-most)
        sim = torch.matmul(x, self.memory.T)
        idx = sim.argmax(dim=-1)

        for b in range(x.size(0)):
            self.memory[idx[b]] = (
                (1 - lr) * self.memory[idx[b]] + lr * x[b]
            )

class FastAdapter(nn.Module):
    def __init__(self, dim=256, num_classes=10):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, context):
        # fuse memory + features
        h = torch.cat([x, context], dim=-1)
        h = self.mlp(h)
        logits = self.classifier(h)
        return logits
    
class HOPEVision(nn.Module):
    def __init__(self, num_classes=10, dim=256):
        super().__init__()

        self.encoder = ViTEncoder(dim)
        self.cms = CMS(dim)
        self.adapter = FastAdapter(dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)

        context, attn = self.cms(features)

        logits = self.adapter(features, context)

        return logits, features, context
    
# Example training loop:
# model = HOPEVision(num_classes=10)

# optimizer_fast = torch.optim.Adam(
#     list(model.encoder.parameters()) +
#     list(model.adapter.parameters()),
#     lr=1e-4
# )

# criterion = nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     for images, labels in dataloader:

#         # -----------------------
#         # FORWARD PASS
#         # -----------------------
#         logits, features, context = model(images)

#         loss = criterion(logits, labels)

#         # -----------------------
#         # FAST UPDATE (every step)
#         # -----------------------
#         optimizer_fast.zero_grad()
#         loss.backward()
#         optimizer_fast.step()

#         # -----------------------
#         # CMS UPDATE (no grad)
#         # -----------------------
#         model.cms.update(features.detach())

# Middle loop update logic
# if step % 50 == 0:
    # @torch.no_grad()
    # def middle_loop_update(model, step):

    #     # 1. Stabilize memory geometry (prevents drift collapse)
    #     model.cms.memory.data = F.normalize(
    #         model.cms.memory.data, dim=-1
    #     )

    #     # 2. Adaptive memory plasticity control
    #     # (reduce update rate if memory is unstable)
    #     sim = torch.matmul(model.cms.memory, model.cms.memory.T)
    #     coherence = sim.mean()

    #     if coherence < 0.2:
    #         model.cms.update_lr *= 0.9   # slow down learning
    #     else:
    #         model.cms.update_lr *= 1.01  # allow adaptation

    #     # 3. Representation anchoring (light regularization)
    #     for p in model.encoder.parameters():
    #         if p.grad is not None:
    #             p.grad *= 0.95  # mild constraint to prevent drift

# Outer loop update logic:
# if epoch % 5 == 0:
    # def outer_loop_update(model, meta_optimizer, val_loader):

    #     # 1. Evaluate current system on validation stream
    #     val_loss = 0.0

    #     for x, y in val_loader:
    #         logits, _, _ = model(x)
    #         val_loss += F.cross_entropy(logits, y)

    #     val_loss /= len(val_loader)

    #     # 2. Meta-objective: improve generalization stability
    #     meta_loss = val_loss + 0.01 * model.cms.memory.norm()

    #     # 3. Update meta-parameters (learning dynamics, not weights)
    #     meta_optimizer.zero_grad()
    #     meta_loss.backward()
    #     meta_optimizer.step()

    #     # 4. Adjust subsystem learning rates (key HOPE idea)
    #     with torch.no_grad():
    #         # slow down fast learner if overfitting detected
    #         if val_loss > model.prev_val_loss:
    #             model.fast_lr *= 0.95
    #             model.cms.update_lr *= 0.95

    #         # otherwise allow more plasticity
    #         else:
    #             model.fast_lr *= 1.01

    #     model.prev_val_loss = val_loss.item()



###### Example full training loop:

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # encoder, cms, adapter assumed defined as before

# class HOPEVision(nn.Module):
#     def __init__(self, encoder, cms, adapter):
#         super().__init__()
#         self.encoder = encoder
#         self.cms = cms
#         self.adapter = adapter

#         self.prev_val_loss = 1e9
#         self.fast_lr = 1e-4
#         self.cms.update_lr = 0.05

# model = HOPEVision(encoder, cms, adapter)

# fast_optimizer = optim.Adam(
#     list(model.encoder.parameters()) +
#     list(model.adapter.parameters()),
#     lr=model.fast_lr
# )

# meta_optimizer = optim.Adam(
#     [], lr=1e-3  # placeholder meta params (can extend later)
# )

# criterion = nn.CrossEntropyLoss()

# MID_STEP = 50     # middle loop frequency
# VAL_EPOCH = 1     # outer loop frequency

# for epoch in range(num_epochs):

#     model.train()

#     for step, (x, y) in enumerate(train_loader):

#         # =========================
#         # FORWARD PASS
#         # =========================
#         features = model.encoder(x)

#         context, _ = model.cms(features)

#         logits = model.adapter(features, context)

#         loss = criterion(logits, y)

#         # =========================
#         # FAST LOOP UPDATE
#         # =========================
#         fast_optimizer.zero_grad()
#         loss.backward()
#         fast_optimizer.step()

#         # =========================
#         # CMS UPDATE (prototype memory)
#         # =========================
#         model.cms.update(features.detach(), lr=model.cms.update_lr)

#         # =========================
#         # MIDDLE LOOP (stability control)
#         # =========================
#         if step % MID_STEP == 0:
#             with torch.no_grad():

#                 # normalize prototypes (prevents drift explosion)
#                 model.cms.memory.data = F.normalize(
#                     model.cms.memory.data, dim=-1
#                 )

#                 # measure memory coherence
#                 sim = torch.matmul(model.cms.memory,
#                                    model.cms.memory.T)
#                 coherence = sim.mean()

#                 # adaptive plasticity control
#                 if coherence < 0.15:
#                     model.cms.update_lr *= 0.9
#                 else:
#                     model.cms.update_lr *= 1.01

#                 # mild encoder stabilization
#                 for p in model.encoder.parameters():
#                     if p.grad is not None:
#                         p.grad *= 0.95

#         # =========================
#         # LOGGING
#         # =========================
#         if step % 100 == 0:
#             print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

#     # =========================
#     # OUTER LOOP (META UPDATE)
#     # =========================
#     model.eval()

#     val_loss = 0.0

#     with torch.no_grad():
#         for x, y in val_loader:

#             feats = model.encoder(x)
#             ctx, _ = model.cms(feats)
#             logits = model.adapter(feats, ctx)

#             val_loss += criterion(logits, y)

#     val_loss /= len(val_loader)

#     meta_loss = val_loss + 0.01 * model.cms.memory.norm()

#     meta_optimizer.zero_grad()
#     meta_loss.backward()
#     meta_optimizer.step()

#     # =========================
#     # ADAPTIVE LEARNING RATE CONTROL
#     # =========================
#     with torch.no_grad():

#         if val_loss > model.prev_val_loss:
#             model.fast_lr *= 0.95
#             model.cms.update_lr *= 0.95
#         else:
#             model.fast_lr *= 1.01
#             model.cms.update_lr *= 1.01

#     # apply updated fast lr
#     for g in fast_optimizer.param_groups:
#         g["lr"] = model.fast_lr

#     model.prev_val_loss = val_loss.item()

#     print(f"\n[OUTER] Epoch {epoch} Val Loss {val_loss.item():.4f}")