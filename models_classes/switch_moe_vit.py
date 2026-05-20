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
from pathlib import Path
import copy
import torch
from torch import nn
import torch.nn.functional as F

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
        routing_alpha: float = 0.4,
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

        # prototypes: separate EMA buffers for mean and std pooled features
        proto_mean = torch.zeros(self.num_experts, self.embed_dim)
        proto_std = torch.zeros(self.num_experts, self.embed_dim)
        # small random init to avoid zero-division
        proto_mean.normal_(mean=0.0, std=0.02)
        proto_std.normal_(mean=0.0, std=0.02)
        self.register_buffer('prototypes_mean', proto_mean)
        self.register_buffer('prototypes_std', proto_std)
        self.ema_alpha = float(ema_alpha)
        self.route_with_cls_token = bool(route_with_cls_token)
        # routing score weighting between mean and std similarities
        self.routing_alpha = float(routing_alpha)

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

        # compute image-level pooled features for routing (mean + std)
        if self.route_with_cls_token:
            # use CLS token as mean, and zero std
            mean_feat = tokens[:, 0]
            std_feat = tokens.new_zeros(mean_feat.shape)
        else:
            patches = tokens[:, 1:, :]
            mean_feat = patches.mean(dim=1)
            std_feat = patches.std(dim=1, unbiased=False)

        # normalized vectors for cosine similarity
        mean_feat_n = F.normalize(mean_feat, dim=-1)
        std_feat_n = F.normalize(std_feat, dim=-1)
        proto_mean_n = F.normalize(self.prototypes_mean, dim=-1)
        proto_std_n = F.normalize(self.prototypes_std, dim=-1)

        sim_mean = torch.matmul(mean_feat_n, proto_mean_n.t())
        sim_std = torch.matmul(std_feat_n, proto_std_n.t())
        scores = self.routing_alpha * sim_mean + (1.0 - self.routing_alpha) * sim_std
        assigned = scores.argmax(dim=1)

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
            return logits, assigned, mean_feat, std_feat
        return logits

    def update_prototypes(self, mean_feats: torch.Tensor, std_feats: torch.Tensor, assigned: torch.Tensor):
        # mean_feats, std_feats: (B, D) assigned: (B,)
        for e in range(self.num_experts):
            mask = (assigned == e)
            if mask.any():
                mean_mean = mean_feats[mask].mean(dim=0)
                mean_std = std_feats[mask].mean(dim=0)
                # EMA update: p <- alpha * p + (1-alpha) * mean_feat
                with torch.no_grad():
                    self.prototypes_mean[e] = self.ema_alpha * self.prototypes_mean[e] + (1.0 - self.ema_alpha) * mean_mean.to(self.prototypes_mean.device)
                    self.prototypes_std[e] = self.ema_alpha * self.prototypes_std[e] + (1.0 - self.ema_alpha) * mean_std.to(self.prototypes_std.device)

    def initialize_prototypes_from_file(self, file_path: str, device: Optional[torch.device] = None, domain_to_expert_map: Optional[dict] = None):
        """Load per-domain-per-class features saved by the helper script and
        initialize model prototypes.

        file format expected: dict(domain_name -> dict(class_name -> {'mean': tensor, 'std': tensor}))
        If number of domains >= num_experts, the first `num_experts` domains
        are used to initialize prototypes. If fewer, remaining experts are
        filled by duplicating the last domain prototype with small noise.
        """
        if device is None:
            device = next(self.parameters()).device

        # load data (support torch.save dict)
        data = torch.load(file_path, map_location='cpu')
        # compute domain-level prototypes by averaging class-level features
        domain_names = list(data.keys())
        domain_means = []
        domain_stds = []
        for dn in domain_names:
            classes = data[dn]
            means = []
            stds = []
            for cls, v in classes.items():
                mv = v.get('mean', None)
                sv = v.get('std', None)
                if isinstance(mv, torch.Tensor):
                    means.append(mv)
                else:
                    means.append(torch.tensor(mv))
                if isinstance(sv, torch.Tensor):
                    stds.append(sv)
                else:
                    stds.append(torch.tensor(sv))
            if len(means) == 0:
                continue
            domain_mean = torch.stack(means, dim=0).mean(dim=0)
            domain_std = torch.stack(stds, dim=0).mean(dim=0)
            domain_means.append(domain_mean)
            domain_stds.append(domain_std)

        if len(domain_means) == 0:
            raise RuntimeError("No domain prototypical features found in file")

        D = self.embed_dim
        # build prototype tensors
        pm = torch.zeros((self.num_experts, D), dtype=torch.float32, device=device)
        ps = torch.zeros((self.num_experts, D), dtype=torch.float32, device=device)

        n_domains = len(domain_means)
        for i in range(min(self.num_experts, n_domains)):
            pm[i] = domain_means[i].to(device=device, dtype=pm.dtype)
            ps[i] = domain_stds[i].to(device=device, dtype=ps.dtype)

        if n_domains < self.num_experts:
            # replicate last domain prototype with tiny noise for remaining experts
            last_m = pm[min(n_domains - 1, 0)].clone()
            last_s = ps[min(n_domains - 1, 0)].clone()
            for i in range(n_domains, self.num_experts):
                pm[i] = last_m + 1e-3 * torch.randn_like(last_m)
                ps[i] = last_s + 1e-3 * torch.randn_like(last_s)

        with torch.no_grad():
            self.prototypes_mean.copy_(pm)
            self.prototypes_std.copy_(ps)

        return domain_names

    def initialize_prototypes_from_dataset(self, dataset_root: str, layer_index: int = 7, device: Optional[torch.device] = None):
        """Walk a folder-structured dataset (domain/class/images) and compute
        per-domain prototypes from the first image of each class. Uses the
        model's patch embedding and lower blocks to compute features at the
        specified layer index.
        Returns the list of domain names used.
        """
        if device is None:
            device = next(self.parameters()).device

        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root {dataset_root} not found")

        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        domain_names = []
        domain_means = []
        domain_stds = []

        for domain_entry in sorted(root.iterdir()):
            if not domain_entry.is_dir():
                continue
            domain_name = domain_entry.name
            class_means = []
            class_stds = []
            for class_entry in sorted(domain_entry.iterdir()):
                if not class_entry.is_dir():
                    continue
                # pick first image
                img_file = None
                for f in sorted(class_entry.iterdir()):
                    if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                        img_file = f
                        break
                if img_file is None:
                    continue
                img = Image.open(img_file).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)

                # forward through patch embed and positional handling
                with torch.inference_mode():
                    x_p = self.patch_embed(x)
                    B = x_p.shape[0]
                    if isinstance(self.cls_token, nn.Parameter):
                        cls = self.cls_token.expand(B, -1, -1).to(device)
                    else:
                        cls = self.cls_token.expand(B, -1, -1).to(device)
                    if self.pos_embed is not None:
                        tokens = torch.cat((cls, x_p), dim=1) + self.pos_embed.to(device)
                    else:
                        tokens = torch.cat((cls, x_p), dim=1)
                    tokens = self.pos_drop(tokens)

                    # apply blocks up to layer_index (use model.blocks if present, else combine lower+upper)
                    # prefer using original timm blocks arrangement if available on model
                    all_blocks = []
                    if hasattr(self, 'lower_blocks') and hasattr(self, 'experts'):
                        # concatenate lower + one expert's upper blocks to form full block list
                        all_blocks = list(self.lower_blocks) + list(self.experts[0].blocks)
                    elif hasattr(self, 'blocks'):
                        all_blocks = list(self.blocks)

                    max_idx = min(layer_index + 1, len(all_blocks))
                    for i in range(max_idx):
                        tokens = all_blocks[i](tokens)

                    patches = tokens[:, 1:, :]
                    mean_feat = patches.mean(dim=1).squeeze(0).cpu()
                    std_feat = patches.std(dim=1, unbiased=False).squeeze(0).cpu()
                    class_means.append(mean_feat)
                    class_stds.append(std_feat)

            if len(class_means) == 0:
                continue
            domain_mean = torch.stack(class_means, dim=0).mean(dim=0)
            domain_std = torch.stack(class_stds, dim=0).mean(dim=0)
            domain_names.append(domain_name)
            domain_means.append(domain_mean)
            domain_stds.append(domain_std)

        if len(domain_means) == 0:
            raise RuntimeError("No prototypes found under dataset root")

        # assign to experts similar to file-based method
        D = self.embed_dim
        pm = torch.zeros((self.num_experts, D), dtype=torch.float32, device=device)
        ps = torch.zeros((self.num_experts, D), dtype=torch.float32, device=device)
        n_domains = len(domain_means)
        for i in range(min(self.num_experts, n_domains)):
            pm[i] = domain_means[i].to(device=device, dtype=pm.dtype)
            ps[i] = domain_stds[i].to(device=device, dtype=ps.dtype)
        if n_domains < self.num_experts:
            last_m = pm[min(n_domains - 1, 0)].clone()
            last_s = ps[min(n_domains - 1, 0)].clone()
            for i in range(n_domains, self.num_experts):
                pm[i] = last_m + 1e-3 * torch.randn_like(last_m)
                ps[i] = last_s + 1e-3 * torch.randn_like(last_s)

        with torch.no_grad():
            self.prototypes_mean.copy_(pm)
            self.prototypes_std.copy_(ps)

        return domain_names

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


def create_switch_moe(num_classes: int = 10, num_experts: int = 4, ema_alpha: float = 0.9, prototype_init_domains: Optional[List[int]] = None, routing_alpha: float = 0.3, **kwargs):
    """Factory returning dict with `model` key to match other factories.

    Parameters beyond the first three are accepted for API compatibility.
    """
    model = SwitchMoEViT(num_classes=num_classes, num_experts=num_experts, ema_alpha=ema_alpha, routing_alpha=routing_alpha)
    return {"model": model}
