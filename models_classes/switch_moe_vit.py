"""
Deterministic Switch-style MoE built on a pretrained ViT (vit_small_patch16_224).

Design notes:
- Shared frozen lower blocks (all but last `upper_blocks`).
- A fixed set of `num_experts` experts, each containing a deepcopy of the
  last `upper_blocks` transformer blocks plus an independent classification head.
- Deterministic switch routing: compute an image-level representation
  (mean of patch embeddings or CLS token) and choose the nearest EMA
  prototype by cosine similarity. All patches of an image are forwarded
  to the chosen expert.
- Prototypes are registered buffers and updated by `update_prototypes`.

The factory function `create_switch_moe` returns a dict compatible with the
existing model factories (key "model").
"""
from typing import Optional, List, Dict, Iterable
import copy
import torch
from torch import nn

try:
    import timm
except Exception:
    timm = None


class ExpertModule(nn.Module):
    def __init__(self, blocks: Iterable[nn.Module], norm: Optional[nn.Module], head: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(b) for b in blocks])
        self.norm = copy.deepcopy(norm) if norm is not None else None
        self.head = head

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N+1, D)
        x = tokens
        for b in self.blocks:
            x = b(x)
        if self.norm is not None:
            x = self.norm(x)
        # classification uses CLS token
        cls = x[:, 0]
        logits = self.head(cls)
        return logits


class SwitchMoEViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_experts: int = 4,
        pretrained_vit: Optional[str] = "small",
        upper_blocks: int = 4,
        ema_alpha: float = 0.9,
        route_with_cls_token: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for SwitchMoEViT: install with `pip install timm`")

        # instantiate a source ViT (small) and split blocks
        # allow strings like 'small' or direct bool
        pretrained_flag = bool(pretrained_vit)
        src = timm.create_model('vit_small_patch16_224', pretrained=pretrained_flag)

        # extract common modules
        self.patch_embed = src.patch_embed
        self.cls_token = src.cls_token if hasattr(src, 'cls_token') else nn.Parameter(torch.zeros(1, 1, src.embed_dim))
        self.pos_embed = src.pos_embed if hasattr(src, 'pos_embed') else None
        self.pos_drop = getattr(src, 'pos_drop', nn.Identity())

        self.embed_dim = getattr(src, 'embed_dim', self.patch_embed.proj.out_channels)
        blocks = list(getattr(src, 'blocks', []))
        n_blocks = len(blocks)
        if upper_blocks <= 0 or upper_blocks > n_blocks:
            raise ValueError(f"upper_blocks must be 1..{n_blocks}")
        self.lower_n = n_blocks - upper_blocks

        # shared frozen lower blocks
        self.lower_blocks = nn.ModuleList(blocks[: self.lower_n])
        # freeze shared backbone
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for b in self.lower_blocks:
            for p in b.parameters():
                p.requires_grad = False
        # also freeze cls_token / pos_embed if they're parameters
        try:
            if isinstance(self.cls_token, nn.Parameter):
                self.cls_token.requires_grad = False
        except Exception:
            pass

        # build experts from upper blocks (deepcopied per-expert)
        upper_blocks_list = blocks[self.lower_n :]
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            head = nn.Linear(self.embed_dim, num_classes)
            # try to copy source head weights when shapes permit
            try:
                if hasattr(src, 'head') and isinstance(src.head, nn.Linear):
                    if src.head.weight.shape == head.weight.shape:
                        head.weight.data.copy_(src.head.weight.data)
                        head.bias.data.copy_(src.head.bias.data)
            except Exception:
                pass
            self.experts.append(ExpertModule(upper_blocks_list, getattr(src, 'norm', None), head))

        # prototypes: EMA buffers (no grad)
        proto = torch.zeros(self.num_experts, self.embed_dim)
        # small random init to avoid zero-division
        proto.normal_(mean=0.0, std=0.02)
        self.register_buffer('prototypes', proto)
        self.ema_alpha = float(ema_alpha)
        self.route_with_cls_token = bool(route_with_cls_token)

        # usage counters for logging (simple single-layer usage)
        self.register_buffer('usage_counts', torch.zeros(self.num_experts, dtype=torch.long))

    def forward(self, x: torch.Tensor, return_assignment: bool = False):
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        # prepend cls token
        if isinstance(self.cls_token, nn.Parameter):
            cls = self.cls_token.expand(B, -1, -1)
        else:
            cls = self.cls_token.expand(B, -1, -1)
        if self.pos_embed is not None:
            # pos_embed shape: (1, N+1, D)
            tokens = torch.cat((cls, x), dim=1) + self.pos_embed
        else:
            tokens = torch.cat((cls, x), dim=1)
        tokens = self.pos_drop(tokens)

        # pass through lower blocks (shared, frozen)
        for b in self.lower_blocks:
            tokens = b(tokens)

        # compute image-level representation for routing
        if self.route_with_cls_token:
            feats = tokens[:, 0]
        else:
            # mean pooling over patch tokens (exclude CLS)
            feats = tokens[:, 1:].mean(dim=1)

        # normalize for cosine similarity
        fn = feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        feats_n = feats / fn
        proto_n = self.prototypes / (self.prototypes.norm(dim=1, keepdim=True).clamp(min=1e-8))
        sims = torch.matmul(feats_n, proto_n.t())  # (B, E)
        assigned = sims.argmax(dim=1)

        # dispatch per-expert (image-level dispatch)
        device = tokens.device
        logits = tokens.new_zeros((B, self.experts[0].head.out_features))
        unique_experts = assigned.unique()
        for e in unique_experts:
            idx = (assigned == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            sub_tokens = tokens[idx]
            out = self.experts[int(e)].forward(sub_tokens)
            logits[idx] = out
            # update usage counts (accumulate; caller may reset per-epoch)
            self.usage_counts[int(e)] += idx.numel()

        if return_assignment:
            return logits, assigned, feats
        return logits

    def update_prototypes(self, feats: torch.Tensor, assigned: torch.Tensor):
        # feats: (B, D) assigned: (B,)
        for e in range(self.num_experts):
            mask = (assigned == e)
            if mask.any():
                mean_feat = feats[mask].mean(dim=0)
                # EMA update: p <- alpha * p + (1-alpha) * mean_feat
                with torch.no_grad():
                    self.prototypes[e] = self.ema_alpha * self.prototypes[e] + (1.0 - self.ema_alpha) * mean_feat.to(self.prototypes.device)

    def get_and_reset_usage_counts(self) -> List[int]:
        counts = self.usage_counts.detach().cpu().tolist()
        self.usage_counts.zero_()
        return [counts]

    def get_router_parameters(self):
        # no learned router
        return []

    def freeze_routing(self, frozen: bool = True):
        # nothing to freeze; provided for API compatibility
        return

    def router_aux_loss(self, cfg: Optional[Dict] = None) -> torch.Tensor:
        return torch.tensor(0.0)


def create_switch_moe(num_classes: int = 10, num_experts: int = 4, ema_alpha: float = 0.9, prototype_init_domains: Optional[List[int]] = None, **kwargs):
    """Factory returning dict with `model` key to match other factories.

    Parameters beyond the first three are accepted for API compatibility.
    """
    model = SwitchMoEViT(num_classes=num_classes, num_experts=num_experts, ema_alpha=ema_alpha)
    return {"model": model}
