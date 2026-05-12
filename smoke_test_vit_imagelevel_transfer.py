"""Smoke test for vit_moe_imagelevel pretrained weight transfer.

Run: python smoke_test_vit_imagelevel_transfer.py

This script creates a timm ViT-Small (`vit_small_patch16_224`) and the
project's `vit_moe_imagelevel` model initialized with `pretrained_vit='small'`.
It then verifies that patch embedding, cls token, pos_embed, per-block norms,
attention in_proj/out_proj (when shapes match), and MLP weights (replicated
into experts) were transferred/copied where shapes permit.
"""
import sys
import traceback

import torch

from models_fcns.model_utils import create_model


def make_cfg():
    return {
        "model": {
            "name": "vit_moe_imagelevel",
            "num_classes": 10,
            "pretrained_vit": "small",
            # use all layers as MoE-capable for testing
            "moe_layer_indices": "all",
            "num_experts": 4,
        }
    }


def main():
    try:
        cfg = make_cfg()
        print("Creating target model (this may download weights via timm)...")
        model = create_model(cfg).eval()
        print("Target model created.")

        # load source timm model
        try:
            import timm
        except Exception:
            print("timm not available; cannot run smoke test that requires pretrained weights.")
            return
        src = timm.create_model('vit_small_patch16_224', pretrained=True)
        src.eval()
        print("Source timm model loaded.")

        # basic checks
        failures = []

        # patch embed
        try:
            s_pe = getattr(src, 'patch_embed', None)
            d_pe = getattr(model, 'patch_embed', None)
            if s_pe is not None and d_pe is not None and hasattr(s_pe, 'proj') and hasattr(d_pe, 'proj'):
                same = torch.allclose(s_pe.proj.weight.detach().cpu(), d_pe.proj.weight.detach().cpu())
                print(f"patch_embed.proj.weight equal: {same}")
                if not same:
                    failures.append('patch_embed')
        except Exception as e:
            print('Error checking patch_embed:', e)
            failures.append('patch_embed')

        # cls token and pos embed
        try:
            if hasattr(src, 'cls_token') and getattr(model, 'cls_token', None) is not None:
                if model.cls_token.shape == src.cls_token.shape:
                    same = torch.allclose(model.cls_token.detach().cpu(), src.cls_token.detach().cpu())
                    print(f"cls_token equal: {same}")
                    if not same:
                        failures.append('cls_token')
            if hasattr(src, 'pos_embed') and getattr(model, 'pos_embed', None) is not None:
                if model.pos_embed.shape == src.pos_embed.shape:
                    same = torch.allclose(model.pos_embed.detach().cpu(), src.pos_embed.detach().cpu())
                    print(f"pos_embed equal: {same}")
                    if not same:
                        failures.append('pos_embed')
        except Exception as e:
            print('Error checking cls/pos:', e)
            failures.append('cls_pos')

        # per-block checks
        n_blocks = min(len(getattr(src, 'blocks', [])), len(getattr(model, 'blocks', [])))
        for i in range(n_blocks):
            sblk = src.blocks[i]
            dblk = model.blocks[i]
            print(f"Checking block {i}...")
            # norms
            try:
                if hasattr(sblk, 'norm1') and hasattr(dblk, 'norm1'):
                    same = torch.allclose(sblk.norm1.weight.detach().cpu(), dblk.norm1.weight.detach().cpu()) and torch.allclose(sblk.norm1.bias.detach().cpu(), dblk.norm1.bias.detach().cpu())
                    print(f"  norm1 equal: {same}")
                    if not same:
                        failures.append(f'norm1_block_{i}')
                if hasattr(sblk, 'norm2') and hasattr(dblk, 'norm2'):
                    same = torch.allclose(sblk.norm2.weight.detach().cpu(), dblk.norm2.weight.detach().cpu()) and torch.allclose(sblk.norm2.bias.detach().cpu(), dblk.norm2.bias.detach().cpu())
                    print(f"  norm2 equal: {same}")
                    if not same:
                        failures.append(f'norm2_block_{i}')
            except Exception as e:
                print('  Error checking norms:', e)
                failures.append(f'norms_block_{i}')

            # attention qkv -> in_proj and out_proj
            try:
                sat = getattr(sblk, 'attn', None)
                dat = getattr(dblk, 'attn', None)
                if sat is not None and dat is not None:
                    # extract src stacked qkv if possible
                    src_qkv_w = None
                    src_qkv_b = None
                    if hasattr(sat, 'qkv'):
                        src_qkv_w = sat.qkv.weight.detach()
                        src_qkv_b = getattr(sat.qkv, 'bias', None)
                        if src_qkv_b is not None:
                            src_qkv_b = src_qkv_b.detach()
                    elif hasattr(sat, 'q') and hasattr(sat, 'k') and hasattr(sat, 'v'):
                        try:
                            src_qkv_w = torch.cat([sat.q.weight.detach(), sat.k.weight.detach(), sat.v.weight.detach()], dim=0)
                            bq = getattr(sat.q, 'bias', None)
                            bk = getattr(sat.k, 'bias', None)
                            bv = getattr(sat.v, 'bias', None)
                            if bq is not None and bk is not None and bv is not None:
                                src_qkv_b = torch.cat([bq.detach(), bk.detach(), bv.detach()], dim=0)
                        except Exception:
                            src_qkv_w = None
                            src_qkv_b = None

                    if src_qkv_w is not None and hasattr(dat, 'in_proj_weight'):
                        tgt_w = dat.in_proj_weight.detach()
                        sw = src_qkv_w.to(device=tgt_w.device, dtype=tgt_w.dtype)
                        equal = False
                        if sw.shape == tgt_w.shape:
                            equal = torch.allclose(sw.cpu(), tgt_w.cpu())
                        else:
                            try:
                                equal = torch.allclose(sw.reshape(tgt_w.shape).cpu(), tgt_w.cpu())
                            except Exception:
                                equal = False
                        print(f"  in_proj_weight equal: {equal}")
                        if not equal:
                            failures.append(f'in_proj_block_{i}')
                    # out proj
                    src_out_w = None
                    src_out_b = None
                    if hasattr(sat, 'proj'):
                        src_out_w = sat.proj.weight.detach()
                        src_out_b = getattr(sat.proj, 'bias', None)
                        if src_out_b is not None:
                            src_out_b = src_out_b.detach()
                    if src_out_w is not None:
                        if hasattr(dat, 'out_proj'):
                            ow = dat.out_proj.weight.detach()
                            sw = src_out_w.to(device=ow.device, dtype=ow.dtype)
                            equal = False
                            if sw.shape == ow.shape:
                                equal = torch.allclose(sw.cpu(), ow.cpu())
                            print(f"  out_proj equal: {equal}")
                            if not equal:
                                failures.append(f'out_proj_block_{i}')
                        elif hasattr(dat, 'proj'):
                            try:
                                ow = dat.proj.weight.detach()
                                sw = src_out_w.to(device=ow.device, dtype=ow.dtype)
                                equal = torch.allclose(sw.cpu(), ow.cpu())
                                print(f"  proj equal: {equal}")
                                if not equal:
                                    failures.append(f'proj_block_{i}')
                            except Exception:
                                failures.append(f'proj_block_{i}')
            except Exception as e:
                print('  Error checking attention:', e)
                failures.append(f'attn_block_{i}')

            # mlp -> experts or mlp
            try:
                if hasattr(sblk, 'mlp') and hasattr(dblk, 'mlp'):
                    sfc1 = getattr(sblk.mlp, 'fc1', None)
                    sfc2 = getattr(sblk.mlp, 'fc2', None)
                    if hasattr(dblk.mlp, 'experts'):
                        # check that each expert has same weights as source fc1/fc2
                        for ei, expert in enumerate(dblk.mlp.experts):
                            ok1 = True
                            ok2 = True
                            if sfc1 is not None and hasattr(expert, 'fc1'):
                                ok1 = torch.allclose(sfc1.weight.detach().cpu(), expert.fc1.weight.detach().cpu())
                            if sfc2 is not None and hasattr(expert, 'fc2'):
                                ok2 = torch.allclose(sfc2.weight.detach().cpu(), expert.fc2.weight.detach().cpu())
                            print(f"  expert {ei} fc1_equal={ok1} fc2_equal={ok2}")
                            if not (ok1 and ok2):
                                failures.append(f'expert_block_{i}_e{ei}')
                    else:
                        if sfc1 is not None and hasattr(dblk.mlp, 'fc1'):
                            ok1 = torch.allclose(sfc1.weight.detach().cpu(), dblk.mlp.fc1.weight.detach().cpu())
                            print(f"  mlp fc1 equal: {ok1}")
                            if not ok1:
                                failures.append(f'mlp_fc1_block_{i}')
                        if sfc2 is not None and hasattr(dblk.mlp, 'fc2'):
                            ok2 = torch.allclose(sfc2.weight.detach().cpu(), dblk.mlp.fc2.weight.detach().cpu())
                            print(f"  mlp fc2 equal: {ok2}")
                            if not ok2:
                                failures.append(f'mlp_fc2_block_{i}')
            except Exception as e:
                print('  Error checking mlp:', e)
                failures.append(f'mlp_block_{i}')

        if failures:
            print('\nSMOKE TEST FAILURES:')
            for f in failures:
                print(' -', f)
            sys.exit(2)
        else:
            print('\nSMOKE TEST PASSED: all checked parameters were copied where shapes matched.')

    except Exception as exc:
        print('Exception during smoke test:')
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
