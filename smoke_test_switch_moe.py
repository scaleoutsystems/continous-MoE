"""Smoke test for the deterministic switch MoE ViT.

Run: python smoke_test_switch_moe.py
"""
import sys
import traceback
import torch

from models_fcns.model_utils import create_model


def make_cfg():
    cfg = {
        "model": {
            "name": "vit_switch_moe",
            "num_classes": 10,
            "pretrained_vit": False,
            "num_experts": 4,
            # architecture fields inherited from the timm ViT used by the factory
        }
    }
    return cfg


def main():
    try:
        cfg = make_cfg()
        print("Creating model from config (may download timm weights)...")
        model = create_model(cfg).eval()
        print("Model created.")

        # simple forward with a random image batch
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print("Forward OK; logits shape:", out.shape)

        # check that expert blocks were copied (weights equal across experts for upper blocks)
        moe_experts = getattr(model, 'experts', None)
        if moe_experts is None:
            print("No experts attribute found on model; skipping expert equality checks.")
            return
        # find a candidate parameter to compare in the first block of upper module
        first = moe_experts[0]
        found = False
        for blk in first.blocks:
            mlp = getattr(blk, 'mlp', None)
            if mlp is not None and hasattr(mlp, 'fc1'):
                w0 = mlp.fc1.weight.detach().cpu()
                eq = True
                for e in moe_experts[1:]:
                    try:
                        w = e.blocks[0].mlp.fc1.weight.detach().cpu()
                        if not torch.allclose(w0, w):
                            eq = False
                            break
                    except Exception:
                        eq = False
                        break
                print(f"Expert upper-MLP fc1 equality across experts: {eq}")
                found = True
                break
        if not found:
            print("Could not find an MLP fc1 in expert blocks to compare; skipping equality check.")

    except Exception as exc:
        print("Exception during smoke test:")
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
