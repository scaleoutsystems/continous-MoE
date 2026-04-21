import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional, Dict, Any
import timm


def create_pretrained_vit_moe_head(**model_cfg) -> nn.Module:
    """Factory returning a PretrainedViTPrototypeMoE instance.

    Expected keys in model_cfg (with defaults):
      - num_classes: int
      - num_experts: int
      - pretrained: bool or torchvision weights object
      - prototype_ema: float (0.99)
      - router_temperature: float (0.1)
      - freeze_encoder: bool (True)
      - use_attraction / use_repulsion / use_consistency: bool
      - use_global_classification: bool
      - loss_weights: dict of weights for attraction/repulsion/consistency/classification
      - repulsion_margin: float
    """
    num_classes = int(model_cfg.get("num_classes", 10))
    num_experts = int(model_cfg.get("num_experts", 4))
    pretrained = model_cfg.get("pretrained", False)
    prototype_ema = float(model_cfg.get("prototype_ema", 0.99))
    router_temperature = float(model_cfg.get("router_temperature", 0.1))
    freeze_encoder = bool(model_cfg.get("freeze_encoder", True))
    use_attraction = bool(model_cfg.get("use_attraction", True))
    use_repulsion = bool(model_cfg.get("use_repulsion", True))
    use_consistency = bool(model_cfg.get("use_consistency", True))
    use_global_classification = bool(model_cfg.get("use_global_classification", False))
    loss_weights = model_cfg.get("loss_weights", {}) or {}
    attraction_weight = float(loss_weights.get("attraction", 1.0))
    repulsion_weight = float(loss_weights.get("repulsion", 1.0))
    consistency_weight = float(loss_weights.get("consistency", 1.0))
    classification_weight = float(loss_weights.get("classification", 1.0))
    repulsion_margin = float(model_cfg.get("repulsion_margin", 0.0))

    return PretrainedViTPrototypeMoE(
        num_classes=num_classes,
        num_experts=num_experts,
        pretrained=pretrained,
        prototype_ema=prototype_ema,
        router_temperature=router_temperature,
        freeze_encoder=freeze_encoder,
        use_attraction=use_attraction,
        use_repulsion=use_repulsion,
        use_consistency=use_consistency,
        use_global_classification=use_global_classification,
        attraction_weight=attraction_weight,
        repulsion_weight=repulsion_weight,
        consistency_weight=consistency_weight,
        classification_weight=classification_weight,
        repulsion_margin=repulsion_margin,
    )


class PretrainedViTPrototypeMoE(nn.Module):
    """Pretrained ViT encoder + prototype-based MoE classification head.

    Routing is done by cosine similarity between normalized encoder features
    and L2-normalized prototype vectors (one prototype per expert). Top-1
    (switch) routing is used: only the selected expert processes each sample.

    The module caches the last forward's features and routing probabilities so
    external code (e.g., training loop) can compute auxiliary losses.
    """

    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        pretrained: Optional[bool] = False, # Currently only pretrained is supported
        prototype_ema: float = 0.99,
        router_temperature: float = 0.1,
        freeze_encoder: bool = True,
        use_attraction: bool = True,
        use_repulsion: bool = True,
        use_consistency: bool = True,
        use_global_classification: bool = False,
        attraction_weight: float = 1.0,
        repulsion_weight: float = 1.0,
        consistency_weight: float = 1.0,
        classification_weight: float = 1.0,
        repulsion_margin: float = 0.0,
    ):
        super().__init__()

        # instantiate torchvision ViT and remove its classifier head so that
        # forward(x) returns a feature vector per sample
        try:
            self.encoder = timm.create_model('vit_small_patch16_128', pretrained=True)
        except Exception:
            self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True, img_size=128)

        # infer feature dim from encoder head if present
        feat_dim = None
        if hasattr(self.encoder, "heads") and isinstance(self.encoder.heads, nn.Linear):
            feat_dim = int(self.encoder.heads.in_features)
        elif hasattr(self.encoder, "head") and isinstance(self.encoder.head, nn.Linear):
            feat_dim = int(self.encoder.head.in_features)
        # replace head so forward returns pre-logit features
        if hasattr(self.encoder, "heads"):
            self.encoder.heads = nn.Identity()
        elif hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()

        if feat_dim is None:
            # fallback to common ViT embedding size
            feat_dim = getattr(self.encoder, "hidden_dim", None) or getattr(self.encoder, "embed_dim", None) or 768
        self.feature_dim = int(feat_dim)

        self.num_experts = int(num_experts)
        self.num_classes = int(num_classes)
        self.prototype_ema = float(prototype_ema)
        self.router_temperature = float(router_temperature)
        self.freeze_encoder = bool(freeze_encoder)

        # expert classifier heads: one linear layer per expert
        self.experts = nn.ModuleList([nn.Linear(self.feature_dim, self.num_classes) for _ in range(self.num_experts)])

        # prototypes (one per expert) stored as buffers (EMA updates, not gradient)
        protos = torch.randn(self.num_experts, self.feature_dim)
        protos = F.normalize(protos, dim=1)
        self.register_buffer("prototypes", protos)

        # usage counters (accumulated across training); also keep epoch-local counts
        self.register_buffer("usage_counts", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("_epoch_usage_counts", torch.zeros(self.num_experts, dtype=torch.long))

        # caches for the last forward pass (detached tensors)
        self._last_features = None
        self._last_gate_probs = None
        self._last_selected = None
        self._last_gate_probs_aug = None

        # router state
        self._frozen = False

        # loss flags and weights
        self.use_attraction = bool(use_attraction)
        self.use_repulsion = bool(use_repulsion)
        self.use_consistency = bool(use_consistency)
        self.use_global_classification = bool(use_global_classification)
        self.attraction_weight = float(attraction_weight)
        self.repulsion_weight = float(repulsion_weight)
        self.consistency_weight = float(consistency_weight)
        self.classification_weight = float(classification_weight)
        self.repulsion_margin = float(repulsion_margin)

    def forward(self, x: torch.Tensor, aug2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        If `aug2` (a second view) is provided, routing probabilities for the
        second view are also computed and stored to support consistency loss.
        """
        feats = self.encoder(x)
        # ensure a 2D feature tensor (B, D)
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)

        feats = F.normalize(feats, dim=1)

        # compute routing similarities and probabilities
        sims = torch.matmul(feats, self.prototypes.t())
        logits_routing = sims / float(self.router_temperature)
        probs = F.softmax(logits_routing, dim=1)
        selected = probs.argmax(dim=1)

        # cache detached values for auxiliary loss computation
        self._last_features = feats.detach()
        self._last_gate_probs = probs.detach()
        self._last_selected = selected.detach()
        self._last_gate_probs_aug = None

        # update usage counters (epoch-local and global)
        with torch.no_grad():
            uniq, cnts = torch.unique(selected, return_counts=True)
            self._epoch_usage_counts[uniq] += cnts.to(self._epoch_usage_counts.device)
            self.usage_counts[uniq] += cnts.to(self.usage_counts.device)

        # prototype EMA updates: update only prototypes of experts that were
        # selected in this batch (use batch mean as an efficient surrogate)
        if not self._frozen:
            with torch.no_grad():
                if 'uniq' in locals() and uniq.numel() > 0:
                    for e_idx in uniq.tolist():
                        mask = selected == int(e_idx)
                        if mask.any():
                            feat_mean = feats[mask].mean(dim=0)
                            p_old = self.prototypes[int(e_idx)]
                            p_new = self.prototype_ema * p_old + (1.0 - self.prototype_ema) * feat_mean
                            p_new = F.normalize(p_new, dim=0)
                            self.prototypes[int(e_idx)] = p_new

        # compute expert logits for all experts then select per-sample
        all_logits = torch.stack([e(feats) for e in self.experts], dim=1)  # (B, E, C)
        batch_idx = torch.arange(feats.size(0), device=feats.device)
        selected_logits = all_logits[batch_idx, selected, :]

        if aug2 is not None:
            # compute gating for the second view too (for consistency loss)
            feats2 = self.encoder(aug2)
            if feats2.ndim > 2:
                feats2 = torch.flatten(feats2, start_dim=1)
            feats2 = F.normalize(feats2, dim=1)
            sims2 = torch.matmul(feats2, self.prototypes.t())
            logits_routing2 = sims2 / float(self.router_temperature)
            probs2 = F.softmax(logits_routing2, dim=1)
            self._last_gate_probs_aug = probs2.detach()

        # choose output logits: by default use the selected-expert logits (Switch)
        if self.use_global_classification:
            # compute weighted sum of expert outputs by routing probability
            global_logits = (probs.unsqueeze(2) * all_logits).sum(dim=1)
            logits = global_logits
        else:
            logits = selected_logits

        return logits

    def router_aux_loss(self, cfg: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute auxiliary router losses based on the last forward pass.

        Returns a scalar tensor (0 if no applicable cached data).
        """
        loss = 0.0
        device = self.prototypes.device

        if self._last_features is None or self._last_gate_probs is None:
            return torch.tensor(0.0, device=device)

        # attraction: encourage selected prototype to be similar to feature
        if self.use_attraction:
            sel = self._last_selected
            # gather prototype for each sample
            prot_for_samples = self.prototypes[sel]
            sim = (self._last_features * prot_for_samples).sum(dim=1)
            attraction = -sim.mean()
            loss = loss + self.attraction_weight * attraction

        # repulsion: encourage prototypes to differ from each other
        if self.use_repulsion:
            P = self.prototypes  # (E, D)
            S = torch.matmul(P, P.t())  # cosine since both normalized
            # zero diagonal
            E = S.shape[0]
            mask = ~torch.eye(E, dtype=torch.bool, device=device)
            off = S[mask]
            if self.repulsion_margin > 0.0:
                off = torch.clamp(off - self.repulsion_margin, min=0.0)
            rep = (off ** 2).mean() if off.numel() > 0 else torch.tensor(0.0, device=device)
            loss = loss + self.repulsion_weight * rep

        # consistency: KL divergence between gating distributions of two views
        if self.use_consistency and self._last_gate_probs_aug is not None:
            p1 = self._last_gate_probs
            p2 = self._last_gate_probs_aug
            # avoid log(0); compute symmetric KL
            p1_log = torch.log(torch.clamp(p1, 1e-9, 1.0))
            p2_log = torch.log(torch.clamp(p2, 1e-9, 1.0))
            kl1 = F.kl_div(p1_log, p2, reduction="batchmean")
            kl2 = F.kl_div(p2_log, p1, reduction="batchmean")
            cons = 0.5 * (kl1 + kl2)
            loss = loss + self.consistency_weight * cons

        return loss

    def router_balance_loss(self, strength: float = 0.1) -> torch.Tensor:
        """Compatibility helper used by compute_aux_loss when router balancing
        is enabled in the config. Computes mean-squared deviation from uniform
        of the last-gating probabilities averaged across the batch.
        """
        if self._last_gate_probs is None:
            return torch.tensor(0.0, device=self.prototypes.device)
        p_mean = self._last_gate_probs.mean(dim=0)
        target = torch.full_like(p_mean, 1.0 / float(max(1, p_mean.numel())))
        loss = ((p_mean - target) ** 2).mean()
        return strength * loss

    # utility hooks expected by the training infrastructure
    def get_router_parameters(self):
        # prototypes are buffers (EMA), router has no trainable parameters
        return []

    def get_and_reset_usage_counts(self):
        vals = self._epoch_usage_counts.detach().cpu().numpy().tolist()
        # reset epoch counts
        self._epoch_usage_counts.zero_()
        return vals

    def freeze_routing(self, freeze: bool = True):
        self._frozen = bool(freeze)

    def adjust_router_learning_rate(self, optimizer, mult: float):
        # nothing to do; router has no optimizer params in this design
        return optimizer
