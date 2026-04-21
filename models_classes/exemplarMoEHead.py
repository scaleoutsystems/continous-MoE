import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ----------------------------
# Backbone (ViT)
# ----------------------------
class ViTBackbone(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, trainable=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if not trainable:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.out_dim = self.vit.num_features

    def forward(self, x):
        return self.vit(x)


# ----------------------------
# MoE Head with Stable EMA
# ----------------------------
class StableEMAMoEHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_classes,
        num_experts=8,
        momentum_fast=0.95,
        momentum_slow=0.995,
        var_floor=1e-2,
        var_ceiling=10.0
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_classes = num_classes

        self.m_fast = momentum_fast
        self.m_slow = momentum_slow

        self.var_floor = var_floor
        self.var_ceiling = var_ceiling

        # experts
        self.experts = nn.ModuleList([
            nn.Linear(feat_dim, num_classes)
            for _ in range(num_experts)
        ])

        # router
        self.router = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_experts)
        )

        # dual EMA stats
        self.register_buffer("mu_fast", torch.zeros(num_experts, feat_dim))
        self.register_buffer("mu_slow", torch.zeros(num_experts, feat_dim))
        self.register_buffer("var", torch.ones(num_experts, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_experts))

    # ----------------------------
    # EMA update (CONFIDENCE GATED)
    # ----------------------------
    @torch.no_grad()
    def update_ema(self, feats, expert_idx, confidence):
        """
        Only update if routing confidence is high.
        """

        for i in range(self.num_experts):
            mask = (expert_idx == i) & (confidence > 0.6)
            if mask.sum() == 0:
                continue

            x = feats[mask]

            batch_mu = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            batch_var = torch.clamp(batch_var, self.var_floor, self.var_ceiling)

            if self.initialized[i] == 0:
                self.mu_fast[i] = batch_mu
                self.mu_slow[i] = batch_mu
                self.var[i] = batch_var
                self.initialized[i] = 1
            else:
                self.mu_fast[i] = self.m_fast * self.mu_fast[i] + (1 - self.m_fast) * batch_mu
                self.mu_slow[i] = self.m_slow * self.mu_slow[i] + (1 - self.m_slow) * batch_mu
                self.var[i] = self.m_slow * self.var[i] + (1 - self.m_slow) * batch_var


# ----------------------------
# Stable W2 proxy distance
# ----------------------------
def gaussian_w2(feats, mu_fast, mu_slow, var):
    """
    Hybrid dual-centroid distance
    """

    x = feats.unsqueeze(1)  # [B,1,D]
    mf = mu_fast.unsqueeze(0)
    ms = mu_slow.unsqueeze(0)
    v = var.unsqueeze(0)

    def dist(mu):
        diff2 = (x - mu) ** 2
        return (diff2 / (v + 1e-6)).sum(dim=-1)

    return 0.5 * dist(mf) + 0.5 * dist(ms)


# ----------------------------
# Domain Shift Detector
# ----------------------------
class DomainShiftDetector(nn.Module):
    def __init__(self, feat_dim, window=128, threshold=2.5):
        super().__init__()
        self.window = window
        self.threshold = threshold

        self.register_buffer("buffer", torch.zeros(window, feat_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1))

        self.register_buffer("ref_mean", torch.zeros(feat_dim))
        self.register_buffer("ref_std", torch.ones(feat_dim))

    @torch.no_grad()
    def update(self, feats):
        B = feats.size(0)

        for i in range(B):
            self.buffer[self.ptr] = feats[i]
            self.ptr[0] = (self.ptr[0] + 1) % self.window
            self.filled[0] = min(self.filled[0] + 1, self.window)

    @torch.no_grad()
    def detect(self, feats):
        """
        returns shift score
        """

        if self.filled[0] < self.window:
            return torch.tensor(0.0, device=feats.device)

        cur_mean = feats.mean(dim=0)

        diff = (cur_mean - self.ref_mean) / (self.ref_std + 1e-6)
        score = torch.norm(diff)

        return score

    @torch.no_grad()
    def update_reference(self):
        if self.filled[0] < self.window:
            return

        data = self.buffer

        self.ref_mean = data.mean(dim=0)
        self.ref_std = data.std(dim=0) + 1e-6


# ----------------------------
# Full MoE Model
# ----------------------------
class StableMoE(nn.Module):
    def __init__(
        self,
        backbone,
        head,
        detector,
        k=2,
        tau=1.0,
        lambda_router=0.6,
        epsilon=0.05
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.detector = detector

        self.k = k
        self.tau = tau
        self.lambda_router = lambda_router
        self.epsilon = epsilon

    def forward(self, x):

        feats = self.backbone(x)

        # update shift buffer
        self.detector.update(feats)

        # shift score
        shift_score = self.detector.detect(feats)

        # adapt routing under shift
        shift_active = shift_score > self.detector.threshold

        router_logits = self.head.router(feats)

        w2 = gaussian_w2(
            feats,
            self.head.mu_fast,
            self.head.mu_slow,
            self.head.var
        )

        metric_logits = -w2 / self.tau

        logits = (1 - self.lambda_router) * router_logits + self.lambda_router * metric_logits

        probs = F.softmax(logits, dim=-1)

        # epsilon exploration (prevents collapse under shift)
        if shift_active:
            probs = (1 - self.epsilon) * probs + self.epsilon / self.head.num_experts

        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)

        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        B = feats.size(0)
        logits_out = torch.zeros(B, self.head.num_classes, device=feats.device)

        for j in range(self.k):
            idx = topk_idx[:, j]
            w = topk_probs[:, j].unsqueeze(-1)

            out = torch.stack([
                self.head.experts[idx[i]](feats[i].unsqueeze(0))
                for i in range(B)
            ], dim=0).squeeze(1)

            logits_out += w * out

        return logits_out, probs, feats, topk_idx, shift_score


# ----------------------------
# Training Step
# ----------------------------
def training_step(model, batch, optimizer, criterion, lambda_balance=0.01):

    x, y = batch

    logits, probs, feats, topk_idx, shift_score = model(x)

    loss_task = criterion(logits, y)

    # balance loss
    avg_probs = probs.mean(dim=0)
    balance_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()

    loss = loss_task + lambda_balance * balance_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # EMA update (ONLY used experts + confidence gating)
    with torch.no_grad():
        confidence = probs.max(dim=-1).values
        model.head.update_ema(feats, topk_idx[:, 0], confidence)

        # update domain reference if stable
        if shift_score < model.detector.threshold:
            model.detector.update_reference()

    return {
        "loss": loss.item(),
        "task_loss": loss_task.item(),
        "balance_loss": balance_loss.item(),
        "shift_score": shift_score.item()
    }


# ----------------------------
# Builder
# ----------------------------
def build_model(num_classes=10, num_experts=8):
    backbone = ViTBackbone(trainable=False)

    head = StableEMAMoEHead(
        feat_dim=backbone.out_dim,
        num_classes=num_classes,
        num_experts=num_experts
    )

    detector = DomainShiftDetector(feat_dim=backbone.out_dim)

    model = StableMoE(
        backbone=backbone,
        head=head,
        detector=detector
    )

    return model