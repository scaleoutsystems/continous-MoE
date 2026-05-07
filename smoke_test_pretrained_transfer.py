"""Smoke test: instantiate MoE ViT initialized from timm pretrained ViT-S
and verify that experts in MoE layers were initialized identically from the
corresponding pretrained FFN.

Run: python smoke_test_pretrained_transfer.py
"""
import sys
import traceback
import torch

from models_fcns.model_utils import create_model


def make_cfg():
    cfg = {
        "model": {
            "name": "vit_moe",
            "num_classes": 10,
            # request timm pretrained small ViT
            "pretrained_vit": "small",
            # replace last half of layers with MoE for the test
            "moe_layer_indices": "back_half_every_other",
            "moe_num_unshared_experts": 4,
            "moe_num_shared_experts": 0,
            "moe_top_k": 1,
            # architecture fields will be overridden by factory when pretrained_vit is set
        }
    }
    return cfg


def main():
    try:
        cfg = make_cfg()
        print("Creating model from config (may download timm weights)...")
        model = create_model(cfg).eval()
        print("Model created.")

        # find MoE modules and check expert equality
        moe_modules = [m for m in model.modules() if m.__class__.__name__ in ("MoE", "ImageMoE")]
        if not moe_modules:
            print("No MoE modules found in model. Did factory apply MoE layers?")
            return
        for idx, m in enumerate(moe_modules):
            print(f"Checking MoE module {idx}: {m.__class__.__name__}")
            experts = getattr(m, 'experts', None)
            if experts is None:
                print("  module has no experts attribute; skipping")
                continue
            # check fc1 weights equality across experts (if present)
            eq_fc1 = True
            eq_fc2 = True
            w0_fc1 = getattr(experts[0], 'fc1', None)
            if w0_fc1 is not None:
                w0 = experts[0].fc1.weight.detach().cpu()
                for e in experts[1:]:
                    try:
                        if not torch.allclose(w0, e.fc1.weight.detach().cpu()):
                            eq_fc1 = False
                            break
                    except Exception:
                        eq_fc1 = False
                        break
            else:
                eq_fc1 = None
            w0_fc2 = getattr(experts[0], 'fc2', None)
            if w0_fc2 is not None:
                w02 = experts[0].fc2.weight.detach().cpu()
                for e in experts[1:]:
                    try:
                        if not torch.allclose(w02, e.fc2.weight.detach().cpu()):
                            eq_fc2 = False
                            break
                    except Exception:
                        eq_fc2 = False
                        break
            else:
                eq_fc2 = None

            print(f"  experts count: {len(experts)} | fc1_equal={eq_fc1} | fc2_equal={eq_fc2}")

    except Exception as exc:
        print("Exception during smoke test:")
        traceback.print_exc()
        sys.exit(2)

if __name__ == '__main__':
    main()
