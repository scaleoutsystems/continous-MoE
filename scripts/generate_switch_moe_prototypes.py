"""Generate mean/std patch features per-domain per-class using timm ViT-S.

Saves a torch file mapping domain -> class -> {'mean': tensor, 'std': tensor}.

Usage:
  python scripts/generate_switch_moe_prototypes.py --dataset-root /path/to/root --out-file prototypes_init.pt

Folder structure expected:
  dataset_root/
    domainA/
      class0/
        img001.jpg
      class1/
    domainB/
      class0/

By default this script processes one image (first) per class.
"""
import argparse
import os
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image

try:
    import timm
except Exception:
    timm = None


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def is_image(fname: str) -> bool:
    return Path(fname).suffix.lower() in IMG_EXTS


def process_image(img_path: str, vit, layer_index: int = 7, device='cpu'):
    vit = vit.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        # patch embed
        x_p = vit.patch_embed(x)  # (1, N, D)
        B = x_p.shape[0]
        if hasattr(vit, 'cls_token'):
            cls = vit.cls_token.expand(B, -1, -1).to(device)
        else:
            cls = torch.zeros((B, 1, x_p.shape[2]), device=device)
        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            tokens = torch.cat((cls, x_p), dim=1) + vit.pos_embed.to(device)
        else:
            tokens = torch.cat((cls, x_p), dim=1)
        tokens = getattr(vit, 'pos_drop', torch.nn.Identity())(tokens)

        # use timm blocks up to layer_index (inclusive)
        blocks = list(getattr(vit, 'blocks', []))
        max_idx = min(layer_index + 1, len(blocks))
        for i in range(max_idx):
            tokens = blocks[i](tokens)

        patches = tokens[:, 1:, :]
        mean_feat = patches.mean(dim=1).squeeze(0).cpu()
        std_feat = patches.std(dim=1, unbiased=False).squeeze(0).cpu()
        return mean_feat, std_feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--out-file', default='prototypes_init.pt')
    p.add_argument('--layer-index', type=int, default=7, help='0-based block index (default 7 -> 8th layer)')
    p.add_argument('--pretrained', action='store_true', help='download timm pretrained ViT')
    args = p.parse_args()

    if timm is None:
        raise RuntimeError('timm is required: pip install timm')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit = timm.create_model('vit_small_patch16_224', pretrained=args.pretrained).to(device)

    root = Path(args.dataset_root)
    out = {}

    for domain_entry in sorted(root.iterdir()):
        if not domain_entry.is_dir():
            continue
        domain_name = domain_entry.name
        out[domain_name] = {}
        for class_entry in sorted(domain_entry.iterdir()):
            if not class_entry.is_dir():
                continue
            class_name = class_entry.name
            # find first image
            img_file = None
            for f in sorted(class_entry.iterdir()):
                if f.is_file() and is_image(str(f)):
                    img_file = f
                    break
            if img_file is None:
                print(f"Warning: no image found in {class_entry}; skipping")
                continue
            mean_feat, std_feat = process_image(str(img_file), vit, layer_index=args.layer_index, device=device)
            out[domain_name][class_name] = {'mean': mean_feat, 'std': std_feat}

    torch.save(out, args.out_file)
    print(f"Saved prototype data to {args.out_file}")


if __name__ == '__main__':
    main()
