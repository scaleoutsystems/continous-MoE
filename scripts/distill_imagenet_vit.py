#!/usr/bin/env python3
"""
Distill a pretrained DeiT3 teacher to ViT students on ImageNet.

Example headless run (1 GPU, torchrun):

nohup python scripts/distill_imagenet_vit.py \
  --student-mlp 8 \
  --epochs 80 \
  --per-gpu-batch 64 \
  --desired-batch 256 \
  --data-path ./data \
  --save-dir ./checkpoints \
  > train.log 2>&1 &

nohup python scripts/distill_imagenet_vit.py \
  --student-mlp 2 \
  --epochs 80 \
  --per-gpu-batch 64 \
  --desired-batch 256 \
  --data-path ./data \
  --save-dir ./checkpoints \
  > train.log 2>&1 &

python scripts/distill_imagenet_vit.py   --student-mlp 2   --epochs 80   --per-gpu-batch 256   --desired-batch 512   --data-path ./datasets   --save-dir ./checkpoints

Requirements:
  pip install torch torchvision timm

This script supports torch.distributed via torchrun. It will validate every 2
epochs and save the best student model (timestamped) and a final checkpoint.
"""

import argparse
import datetime
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import datasets as hf_datasets
import numpy as np
from PIL import Image
import io

import matplotlib.pyplot as plt

try:
    # timm provides models and Mixup helper
    import timm
    from timm.data.mixup import Mixup
except Exception:
    raise RuntimeError("Please install timm: pip install timm")


def parse_args():
    p = argparse.ArgumentParser(description="Distill DeiT3->ViT on ImageNet")
    p.add_argument("--data-path", default="/datasets", help="root data path")
    p.add_argument("--student-mlp", type=int, choices=[8, 2], default=8)
    p.add_argument("--per-gpu-batch", type=int, default=None,
                   help="mini-batch per GPU (will be overridden by defaults)")
    p.add_argument("--desired-batch", type=int, default=None,
                   help="effective desired batch size (across GPUs)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None, help="KL weight")
    p.add_argument("--temp", type=float, default=None, help="KD temperature")
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="resume checkpoint path")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--hf-id", default="clane9/imagenet-100", help="Hugging Face dataset id to use (default: imagenet-100)")
    p.add_argument("--hf-cache-name", default="imagenet100_hf", help="local folder name under data-path to cache HF dataset")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()


def setup_distributed(args):
    # torchrun sets WORLD_SIZE and LOCAL_RANK
    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.world_size = 1
        args.rank = 0
    return device


def seed_all(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    # ImageNet mean/std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


class HFImageDataset(TorchDataset):
    """A thin wrapper around a Hugging Face image dataset to expose a PyTorch Dataset API."""
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

        # detect image and label column names
        self.image_col = None
        self.label_col = None
        for k, f in self.ds.features.items():
            tname = type(f).__name__
            if tname == 'Image' and self.image_col is None:
                self.image_col = k
            if tname in ('ClassLabel', 'Value') and self.label_col is None:
                # ClassLabel is preferred
                if tname == 'ClassLabel' or k.lower() in ('label', 'labels', 'class'):
                    self.label_col = k

        # fallbacks
        if self.image_col is None:
            # try common names
            for cand in ('image', 'img', 'pixel'):
                if cand in self.ds.column_names:
                    self.image_col = cand
                    break
        if self.label_col is None:
            for cand in ('label', 'labels', 'class', 'label_id'):
                if cand in self.ds.column_names:
                    self.label_col = cand
                    break

        if self.image_col is None:
            raise RuntimeError('Could not find image column in HF dataset; columns: ' + str(self.ds.column_names))
        if self.label_col is None:
            # allow unlabeled datasets (not ideal)
            self.label_col = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = item[self.image_col]
        # HF Image feature may be PIL.Image.Image or numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        elif isinstance(img, dict) and 'bytes' in img:
            img = Image.open(io.BytesIO(img['bytes']))

        # FORCE RGB HERE
        if isinstance(img, Image.Image):
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.label_col is None:
            return img, -1
        lbl = item[self.label_col]
        # ClassLabel features may return int directly
        if isinstance(lbl, dict) and 'label' in lbl:
            lbl = lbl['label']
        return img, int(lbl)


def _load_or_cache_hf_imagenet(hf_id, data_root, cache_name, is_main):
    save_root = os.path.join(data_root, cache_name)
    train_cache = os.path.join(save_root, 'train')
    val_cache = os.path.join(save_root, 'validation')

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        if is_main:
            print(f"Loading cached HF dataset from {save_root}")
        train_ds = hf_datasets.load_from_disk(train_cache)
        val_ds = hf_datasets.load_from_disk(val_cache)
        return train_ds, val_ds

    if is_main:
        print(f"Downloading Hugging Face dataset '{hf_id}' and caching to {save_root} (this may take a while)")

    # Attempt to load dataset; try multiple split keys if needed
    ds = hf_datasets.load_dataset("clane9/imagenet-100")
    # ds is usually a DatasetDict
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        # prefer 'train' and 'validation' splits
        train_ds = ds.get('train') if 'train' in ds else None
        val_ds = ds.get('validation') if 'validation' in ds else None
        if val_ds is None:
            val_ds = ds.get('val') if 'val' in ds else None
        if train_ds is None:
            # fallback to first split
            first_key = list(ds.keys())[0]
            train_ds = ds[first_key]
            # try to get second split as val
            if len(ds.keys()) > 1:
                second_key = list(ds.keys())[1]
                val_ds = ds[second_key]
    else:
        train_ds = ds
        val_ds = None

    # If no validation split, carve out ~50k images
    if val_ds is None:
        total = len(train_ds)
        test_size = max(1, int(0.1 * total))
        split = train_ds.train_test_split(test_size=test_size, shuffle=True, seed=42)
        train_ds = split['train']
        val_ds = split['test']

    # save caches
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(val_cache, exist_ok=True)
    train_ds.save_to_disk(train_cache)
    val_ds.save_to_disk(val_cache)

    return train_ds, val_ds


def make_dataloaders(data_path, per_gpu_bs, num_workers, distributed, hf_id='imagenet-100', hf_cache_name='imagenet100_hf'):
    """Prefer Hugging Face imagenet-100 (cached under data_path/hf_cache_name)."""
    train_t, val_t = build_transforms()

    is_main = (int(os.environ.get('RANK', '0')) == 0)
    train_dataset = []
    val_dataset = []

    try:
        train_hf, val_hf = _load_or_cache_hf_imagenet(hf_id, data_path, hf_cache_name, is_main)
        train_dataset = HFImageDataset(train_hf, transform=train_t)
        val_dataset = HFImageDataset(val_hf, transform=val_t)
        if is_main:
            print(f"Using Hugging Face dataset '{hf_id}' with {len(train_dataset)} train and {len(val_dataset)} val samples")
    except Exception as e:
        if is_main:
            print(f"Failed to load HF dataset '{hf_id}': {e}.")
            raise e

    train_loader = DataLoader(train_dataset, batch_size=per_gpu_bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=per_gpu_bs, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def create_teacher(device):
    teacher = timm.create_model('deit3_small_patch16_224', pretrained=True, num_classes=1000)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def create_student(mlp_ratio):
    # create a ViT and override depth/heads/embed and mlp_ratio
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=1000,
                              embed_dim=192, depth=12, num_heads=3, mlp_ratio=mlp_ratio)
    return model


def compute_ce_loss(outputs, targets):
    # supports hard labels (long) and soft targets (one-hot floats)
    if targets.dtype == torch.long:
        return F.cross_entropy(outputs, targets)
    else:
        logp = F.log_softmax(outputs, dim=1)
        return -(targets * logp).sum(dim=1).mean()


def validate(model, val_loader, device, distributed):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, target in val_loader:
            imgs = imgs.to(device)
            target = target.to(device)
            out = model(imgs)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += imgs.size(0)

    if distributed:
        t = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct, total = int(t[0].item()), int(t[1].item())

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


def save_model(student, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"student_{tag}_{ts}.pth"
    path = os.path.join(save_dir, fname)
    to_save = student.module.state_dict() if isinstance(student, torch.nn.parallel.DistributedDataParallel) else student.state_dict()
    torch.save(to_save, path)
    return path


def reduce_mean(tensor, world_size):
    if world_size <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def main():
    args = parse_args()
    # set sensible defaults depending on student type
    if args.student_mlp == 8:
        defaults = dict(per_gpu_batch=256, desired_batch=512, epochs=80,
                        lr=1e-3, weight_decay=0.05, alpha=0.4, temp=2.0)
    else:
        defaults = dict(per_gpu_batch=256, desired_batch=256, epochs=100,
                        lr=5e-4, weight_decay=0.04, alpha=0.7, temp=2.5)

    if args.per_gpu_batch is None:
        args.per_gpu_batch = defaults['per_gpu_batch']
    if args.desired_batch is None:
        args.desired_batch = defaults['desired_batch']
    if args.epochs is None:
        args.epochs = defaults['epochs']
    if args.lr is None:
        args.lr = defaults['lr']
    if args.weight_decay is None:
        args.weight_decay = defaults['weight_decay']
    if args.alpha is None:
        args.alpha = defaults['alpha']
    if args.temp is None:
        args.temp = defaults['temp']

    seed_all(args.seed)
    device = setup_distributed(args)

    is_main = (args.rank == 0)
    if is_main:
        print("Arguments:", args)

    # dataloaders (prefer HF imagenet-100 cached under data-path)
    train_loader, val_loader = make_dataloaders(
        args.data_path, args.per_gpu_batch, args.num_workers, args.distributed,
        hf_id=args.hf_id, hf_cache_name=args.hf_cache_name)

    # compute accumulation steps so effective batch approximates desired_batch
    world_bs = args.per_gpu_batch * args.world_size
    accum_steps = max(1, math.ceil(args.desired_batch / max(1, world_bs)))
    if is_main:
        print(f"per-gpu-batch={args.per_gpu_batch}, world_size={args.world_size}, accum_steps={accum_steps}")

    teacher = create_teacher(device)
    student = create_student(mlp_ratio=args.student_mlp)
    student.to(device)

    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # total optimization updates (per-step updates, taking accumulation into account)
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(0.05 * total_updates))

    def lr_lambda(step):
        if step < warmup_updates:
            return float(step) / float(max(1, warmup_updates))
        else:
            progress = float(step - warmup_updates) / float(max(1, total_updates - warmup_updates))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.grad_scaler.GradScaler()

    # Mixup
    mixup_fn = None
    if not args.no_mixup:
        mixup_fn = Mixup(mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                         mode='batch', label_smoothing=0.1, num_classes=1000)

    best_val = -1.0
    global_update = 0

    val_history = []
    val_epochs = []

    for epoch in range(args.epochs):

        student.train()
        epoch_loss = 0.0
        if is_main:
            start_time = time.time()

        for it, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if mixup_fn is not None:
                images, target = mixup_fn(images, target)

            with torch.autocast(device_type='cuda', enabled=True):
                student_logits = student(images)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                loss_ce = compute_ce_loss(student_logits, target)
                # KD (logit matching)
                T = args.temp
                kd = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                              F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T * T)

                loss = (1.0 - args.alpha) * loss_ce + args.alpha * kd
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_update += 1

            epoch_loss += loss.item() * accum_steps

        if is_main:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} train_loss={epoch_loss/len(train_loader):.4f} time={elapsed:.1f}s")

        # Validate every 4 epochs
        if (((epoch + 1) % 4) == 0 and (epoch+1 > 60)):
            val_acc = validate(student, val_loader, device, args.distributed)

            if is_main:
                print(f"Validation @ epoch {epoch+1}: acc={val_acc:.4f}")
                val_history.append(val_acc)
                val_epochs.append(epoch + 1)

            if is_main and val_acc > best_val:
                best_val = val_acc
                tag = f"mlp{args.student_mlp}_best"
                path = save_model(student, args.save_dir, tag)
                print(f"Saved best student to {path}")

    # finished training: save final
    if is_main:
        tag = f"mlp{args.student_mlp}_final"
        path = save_model(student, args.save_dir, tag)
        print(f"Saved final student to {path}")

    if is_main and len(val_history) > 0:
        os.makedirs(args.save_dir, exist_ok=True)

        plt.figure()
        plt.plot(val_epochs, val_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Validation Accuracy over Epochs")
        plt.grid(True)

        graph_path = os.path.join(args.save_dir, f"val_acc_mlp{args.student_mlp}.png")
        plt.savefig(graph_path)
        plt.close()

        print(f"Saved validation accuracy plot to {graph_path}")


if __name__ == '__main__':
    main()
