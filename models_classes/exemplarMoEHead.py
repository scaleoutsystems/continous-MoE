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
# MoE Head with EMA Experts
# ----------------------------
class EMAMoEHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_classes,
        num_experts=8,
        momentum=0.99,
        eps=1e-6
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_classes = num_classes
        self.momentum = momentum
        self.eps = eps

        # expert classifiers
        self.experts = nn.ModuleList([
            nn.Linear(feat_dim, num_classes)
            for _ in range(num_experts)
        ])

        # learned router
        self.router = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_experts)
        )

        # EMA statistics per expert
        self.register_buffer("centroids", torch.zeros(num_experts, feat_dim))
        self.register_buffer("var", torch.ones(num_experts, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_experts))

    @torch.no_grad()
    def update_ema(self, feats, expert_idx):
        """
        ONLY update experts that were actually used.

        feats: [B, D]
        expert_idx: [B] (top-1 routing indices)
        """

        for i in range(self.num_experts):
            mask = (expert_idx == i)
            if mask.sum() == 0:
                continue  # IMPORTANT: only update used experts

            x = feats[mask]

            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False) + self.eps

            if self.initialized[i] == 0:
                self.centroids[i] = batch_mean
                self.var[i] = batch_var
                self.initialized[i] = 1
            else:
                m = self.momentum
                self.centroids[i] = m * self.centroids[i] + (1 - m) * batch_mean
                self.var[i] = m * self.var[i] + (1 - m) * batch_var


# ----------------------------
# Gaussian W2 proxy distance AKA Mahalanobis w/ log variance
# ----------------------------
def gaussian_w2(feats, centroids, var):
    """
    feats: [B, D]
    centroids: [E, D]
    var: [E, D]

    returns: [B, E]
    """

    x = feats.unsqueeze(1)        # [B,1,D]
    mu = centroids.unsqueeze(0)   # [1,E,D]
    v = var.unsqueeze(0)          # [1,E,D]

    diff2 = (x - mu) ** 2
    mahal = diff2 / (v + 1e-6)
    logvar = torch.log(v + 1e-6)

    return (mahal + logvar).sum(dim=-1)  # [B,E]


# ----------------------------
# Full MoE Model
# ----------------------------
class MoEModel(nn.Module):
    def __init__(
        self,
        backbone,
        head,
        k=2,
        tau=1.0,
        lambda_router=0.5
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        self.k = k
        self.tau = tau
        self.lambda_router = lambda_router

    def forward(self, x):
        """
        Returns:
            logits: final prediction
            probs: routing probabilities
            feats: backbone features
            topk_idx: selected experts
        """

        feats = self.backbone(x)  # [B,D]

        # learned router logits
        router_logits = self.head.router(feats)  # [B,E]

        # W2 proxy distances
        w2 = gaussian_w2(
            feats,
            self.head.centroids,
            self.head.var
        )

        metric_logits = -w2 / self.tau

        # annealed mixture of learned + metric routing
        logits = (
            (1 - self.lambda_router) * router_logits +
            self.lambda_router * metric_logits
        )

        probs = F.softmax(logits, dim=-1)

        # top-k routing
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)

        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # expert outputs
        B = feats.shape[0]
        outputs = []

        for j in range(self.k):
            idx = topk_idx[:, j]
            weight = topk_probs[:, j].unsqueeze(-1)

            out = []
            for i in range(B):
                expert_id = idx[i].item()
                out_i = self.head.experts[expert_id](feats[i].unsqueeze(0))
                out.append(out_i)

            out = torch.cat(out, dim=0)  # [B, C]
            outputs.append(weight * out)

        logits_out = torch.stack(outputs, dim=0).sum(dim=0)

        return logits_out, probs, feats, topk_idx


# ----------------------------
# Training Step
# ----------------------------
def training_step(model, batch, optimizer, criterion, lambda_balance=0.01):
    """
    batch: (x, y)
    """

    x, y = batch

    logits, probs, feats, topk_idx = model(x)

    # task loss
    task_loss = criterion(logits, y)

    # load balancing (prevents expert collapse)
    avg_probs = probs.mean(dim=0)
    balance_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()

    loss = task_loss + lambda_balance * balance_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ----------------------------
    # EMA UPDATE (ONLY USED EXPERTS)
    # ----------------------------
    with torch.no_grad():
        # use top-1 expert for EMA updates (important for stability)
        model.head.update_ema(feats, topk_idx[:, 0])

    return {
        "loss": loss.item(),
        "task_loss": task_loss.item(),
        "balance_loss": balance_loss.item()
    }


# ----------------------------
# Example construction
# ----------------------------
def build_model(num_classes=10, num_experts=8):
    backbone = ViTBackbone(trainable=False)
    head = EMAMoEHead(
        feat_dim=backbone.out_dim,
        num_classes=num_classes,
        num_experts=num_experts
    )

    model = MoEModel(
        backbone=backbone,
        head=head,
        k=2,
        tau=1.0,
        lambda_router=0.5
    )

    return model