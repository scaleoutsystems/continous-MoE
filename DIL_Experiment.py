#!/usr/bin/env python3
"""Scripted version of the DIL experiment (headless, no plotting).

Usage:
  python DIL_Experiment.py --config configs/example_config.jsonc [--training-seed 123 ...]

If any seed flags or a seeds file/json are provided, the seeds in the config
are ignored and replaced with the provided seed mapping (missing seed types
are treated as random/unspecified).

The script saves the logger output (including epoch_confusion_matrices and
the resolved seeds) into the results folder as the notebook version does.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime

import torch
import numpy as np
import random

from data_analysis_fcns.DIL_Logger import DIL_Logger
from dataset_fcns.replay import ReplayBuffer
from experiment_fcns.setup import load_config
from experiment_fcns.experiment_utils import (
    evaluate_accuracy,
    _maybe_update_router,
    evaluate_full_test,
    _make_scheduler_for_optimizer,
    _make_optimizer_for_model,
    compute_aux_loss,
)

from copy import deepcopy

SEED_KEYS = [
    "global",
    "dataset",
    "partition",
    "loader",
    "model",
    "training",
    "pretrain",
    "baseline",
    "replay",
]


def read_jsonc(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r"//.*", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    return json.loads(raw)


def write_temp_cfg(cfg_dict):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonc", mode="w", encoding="utf-8")
    json.dump(cfg_dict, tf, indent=2)
    tf.flush()
    tf.close()
    return tf.name


def parse_cli_seeds(args):
    mapping = {}
    for k in SEED_KEYS:
        v = getattr(args, f"{k.replace('-', '_')}_seed", None)
        if v is not None:
            mapping[k] = v
    # support --seeds-json
    if args.seeds_json:
        parsed = json.loads(args.seeds_json)
        mapping.update(parsed)
    # support --seeds-file
    if args.seeds_file:
        parsed = read_jsonc(args.seeds_file)
        mapping.update(parsed)
    # normalize values: accept 'random' string or integers
    norm = {}
    for k, v in mapping.items():
        if isinstance(v, str) and v.lower() == "random":
            norm[k] = "random"
        elif v is None:
            norm[k] = None
        else:
            try:
                norm[k] = int(v)
            except Exception:
                norm[k] = v
    return norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to experiment config (JSONC)")
    p.add_argument("--seeds-file", help="Path to JSON/JSONC file with seed mapping")
    p.add_argument("--seeds-json", help="Seed mapping as JSON string")
    # per-seed overrides
    for k in SEED_KEYS:
        p.add_argument(f"--{k}-seed", dest=f"{k}_seed", help=f"Seed value for {k} (int or 'random')")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader num_workers (writes to temp config)")
    p.add_argument("--save-dir", default=None, help="Optional save dir override")

    args = p.parse_args()

    # read original config (keep comments removed)
    cfg_dict = read_jsonc(args.config)

    cli_seeds = parse_cli_seeds(args)
    use_cli_seeds = len(cli_seeds) > 0

    # if CLI provided num_workers, set it in cfg (so loaders honor it)
    if args.num_workers is not None:
        cfg_dict["num_workers"] = int(args.num_workers)

    temp_cfg_path = None
    try:
        if use_cli_seeds:
            # replace seeds in the config entirely with CLI-specified mapping
            new_seeds = {k: None for k in SEED_KEYS}
            for k, v in cli_seeds.items():
                new_seeds[k] = v
            cfg_dict["seeds"] = new_seeds
            temp_cfg_path = write_temp_cfg(cfg_dict)
            cfg_path_to_load = temp_cfg_path
        else:
            cfg_path_to_load = args.config

        setup = load_config(cfg_path_to_load)
        cfg = setup["cfg"]
        device = setup["device"]
        train_loaders = setup["train_loaders"]
        test_loaders = setup["test_loaders"]
        pretrain_loader = setup.get("pretrain_loader", None)
        dataset = setup.get("dataset")

        num_classes = cfg.get("model", {}).get("num_classes", 10)
        num_domains = len(train_loaders)
        epochs_per_domain = cfg.get("epochs_per_domain", 5)

        # build logger and attach resolved seeds for saving
        baseline_val = None
        if cfg.get("baseline", {}).get("train", False):
            baseline_val = None
        logger = DIL_Logger(N=num_domains, C=num_classes, baseline=baseline_val, config_file=args.config)
        logger.seeds = setup.get("resolved_seeds", None)

        # replay buffer
        replay_cfg = cfg.get("replay", {})
        replay_buffer = None
        if replay_cfg.get("enabled", False):
            replay_buffer = ReplayBuffer(capacity=replay_cfg.get("capacity", 1000), policy=replay_cfg.get("policy", "fifo"))

        # model/optimizer/criterion from setup
        model = setup["model"].to(device)
        optimizer = setup["optimizer"]
        criterion = setup["criterion"]
        if hasattr(criterion, "to"):
            criterion = criterion.to(device)

        scheduler = setup.get("scheduler", None)
        pretrain_scheduler = setup.get("pretrain_scheduler", None)

        total_epochs = 0

        # training state
        global_step = 0
        global_epoch = 0
        router_frozen = False

        # Pretraining stage (if enabled)
        try:
            pretrain_spec = cfg.get("seeds", {}).get("pretrain") if cfg is not None else None
            pretrain_seed = setup.get("resolved_seeds", {}).get("pretrain") if setup.get("resolved_seeds") is not None else None
        except Exception:
            pretrain_spec = None
            pretrain_seed = None

        if cfg.get("pretrain", {}).get("enabled", False) and pretrain_loader is not None:
            if pretrain_seed is not None:
                torch.manual_seed(pretrain_seed)
                np.random.seed(pretrain_seed)
                random.seed(pretrain_seed)
                print(f"Applied pretrain seed: {pretrain_seed}")
            elif isinstance(pretrain_spec, str) and str(pretrain_spec).lower() == "random":
                rand_seed = int.from_bytes(os.urandom(8), "big") % (2 ** 32 - 1)
                torch.manual_seed(rand_seed)
                np.random.seed(rand_seed)
                random.seed(rand_seed)
                print("Using randomized seed for pretrain stage")

            print("===== Pretraining stage =====")
            epochs_pretrain = cfg.get("pretrain", {}).get("epochs", epochs_per_domain)
            for epoch in range(epochs_pretrain):
                model.train()
                for x, y in pretrain_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    if replay_buffer is not None:
                        rx, ry = replay_buffer.sample(y.size(0))
                        if rx.numel() > 0:
                            rx, ry = rx.to(device), ry.to(device)
                            rlogits = model(rx)
                            loss = loss + criterion(rlogits, ry)
                    aux = compute_aux_loss(model, cfg)
                    if isinstance(aux, torch.Tensor):
                        loss = loss + aux
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    if replay_buffer is not None:
                        replay_buffer.add_batch(x.cpu(), y.cpu())

                if pretrain_scheduler is not None:
                    pretrain_scheduler.step()

                optimizer, router_frozen = _maybe_update_router(cfg, model, optimizer, pretrain_scheduler, global_epoch, router_frozen)
                global_epoch += 1

            # after pretraining evaluate once
            pretrain_acc, pretrain_conf = evaluate_full_test(model, test_loaders, device, num_classes)
            print(f"Pretraining complete ({epochs_pretrain} epochs). Full test acc: {pretrain_acc:.4f}")
            logger.pretrain_confusion = pretrain_conf

        # Baseline models from post-pretrain weights
        if cfg.get("baseline", {}).get("train", False):
            print("===== Baseline model stage =====")
            post_pretrain_state = model.state_dict()
            baseline_accs = []
            freeze_router_flag = cfg.get("router_freeze_after_epochs") is not None or cfg.get("router_freeze_after_batches") is not None
            multiplier_defined = cfg.get("router_lr_multiplier") is not None
            use_router_balancing = cfg.get("router_balancing", False) and not (freeze_router_flag or multiplier_defined)

            # baseline seed handling
            baseline_spec = cfg.get("seeds", {}).get("baseline") if cfg is not None else None
            baseline_seed = setup.get("resolved_seeds", {}).get("baseline") if setup.get("resolved_seeds") is not None else None
            if baseline_seed is not None:
                torch.manual_seed(baseline_seed)
                np.random.seed(baseline_seed)
                random.seed(baseline_seed)
                print(f"Applied baseline seed: {baseline_seed}")
            elif isinstance(baseline_spec, str) and str(baseline_spec).lower() == "random":
                rand_seed = int.from_bytes(os.urandom(8), "big") % (2 ** 32 - 1)
                torch.manual_seed(rand_seed)
                np.random.seed(rand_seed)
                random.seed(rand_seed)
                print("Using randomized seed for baseline stage")

            for domain_id in range(num_domains):
                bmodel = deepcopy(model)
                bmodel.to(device)
                try:
                    missing, unexpected = bmodel.load_state_dict(post_pretrain_state, strict=False)
                except Exception:
                    bmodel.load_state_dict(post_pretrain_state)
                boptimizer = _make_optimizer_for_model(bmodel, cfg=cfg)
                bscheduler = _make_scheduler_for_optimizer(boptimizer, cfg=cfg, epochs_per_domain=epochs_per_domain)

                if freeze_router_flag and hasattr(bmodel, "freeze_routing"):
                    bmodel.freeze_routing(True)
                if multiplier_defined and hasattr(bmodel, "adjust_router_learning_rate"):
                    boptimizer = bmodel.adjust_router_learning_rate(boptimizer, cfg.get("router_lr_multiplier", 1.0))

                for epoch in range(epochs_per_domain):
                    bmodel.train()
                    for x, y in train_loaders[domain_id]:
                        x, y = x.to(device), y.to(device)
                        boptimizer.zero_grad()
                        logits = bmodel(x)
                        loss_b = criterion(logits, y)
                        if use_router_balancing:
                            aux_b = compute_aux_loss(bmodel, cfg)
                            if isinstance(aux_b, torch.Tensor):
                                loss_b = loss_b + aux_b
                        loss_b.backward()
                        boptimizer.step()
                    if bscheduler is not None:
                        bscheduler.step()

                acc = evaluate_accuracy(bmodel, test_loaders[domain_id], device)
                full_acc, full_conf = evaluate_full_test(bmodel, test_loaders, device, num_classes)
                baseline_accs.append(acc)
                if not hasattr(logger, "baseline_confusions"):
                    logger.baseline_confusions = []
                logger.baseline_confusions.append(full_conf)
                del bmodel, boptimizer, bscheduler
                torch.cuda.empty_cache()
            print("Baseline accuracies per domain:", baseline_accs)
            logger.baseline = baseline_accs

        # Apply training seed after pretraining/baseline as requested
        training_spec = cfg.get("seeds", {}).get("training") if cfg is not None else None
        training_seed = setup.get("resolved_seeds", {}).get("training") if setup.get("resolved_seeds") is not None else None
        if training_seed is not None:
            torch.manual_seed(training_seed)
            np.random.seed(training_seed)
            random.seed(training_seed)
            print(f"Applied training seed: {training_seed}")
        elif isinstance(training_spec, str) and str(training_spec).lower() == "random":
            rand_seed = int.from_bytes(os.urandom(8), "big") % (2 ** 32 - 1)
            torch.manual_seed(rand_seed)
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            print("Using randomized seed for training stage")

        # Main continual training

        # recreate scheduler in case the learning groups changed due to routerLR mult or otherwise
        scheduler = _make_scheduler_for_optimizer(optimizer, cfg=cfg, epochs_per_domain=epochs_per_domain)

        print("===== Main model stage =====")
        for domain_id in range(num_domains):
            print(f"\n===== Training Domain {domain_id} =====")
            train_loader = train_loaders[domain_id]

            # update loss weights if supported
            if hasattr(criterion, "update_weights"):
                try:
                    indices = train_loader.dataset.indices
                    base = train_loader.dataset.dataset
                except Exception:
                    indices = None
                    base = train_loader.dataset
                if indices is not None:
                    labels = [base[i][1] for i in indices]
                    labels = np.array(labels)
                else:
                    labs = []
                    for _, y in train_loader:
                        labs.append(y.cpu().numpy())
                    labels = np.concatenate(labs) if labs else np.array([])
                counts = np.bincount(labels, minlength=num_classes)
                criterion.update_weights(counts, cumulative=getattr(criterion, "_weight_cumulative", False))
                if hasattr(criterion, "weights"):
                    print(f"Updated loss weights for domain {domain_id}: {criterion.weights}")

            for epoch in range(epochs_per_domain):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    if replay_buffer is not None:
                        rx, ry = replay_buffer.sample(y.size(0))
                        if rx.numel() > 0:
                            rx, ry = rx.to(device), ry.to(device)
                            rlogits = model(rx)
                            loss = loss + criterion(rlogits, ry)

                    aux = compute_aux_loss(model, cfg)
                    if isinstance(aux, torch.Tensor):
                        loss = loss + aux
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    if replay_buffer is not None:
                        replay_buffer.add_batch(x.cpu(), y.cpu())

                if scheduler is not None:
                    scheduler.step()

                optimizer, router_frozen = _maybe_update_router(cfg, model, optimizer, scheduler, global_epoch, router_frozen)

                domain_acc, overall_acc, preds, targets, expert_usage = logger.evaluate(model, test_loaders, device)
                # collect routing temperatures from model submodules (if any)
                routing_temps = []
                routing_temps_init = []
                try:
                    for m in model.modules():
                        if hasattr(m, "routing_temp") or hasattr(m, "routing_temp_init"):
                            rt = getattr(m, "routing_temp", None)
                            rti = getattr(m, "routing_temp_init", None)
                            routing_temps.append(None if rt is None else float(rt))
                            routing_temps_init.append(None if rti is None else float(rti))
                except Exception:
                    routing_temps = []
                    routing_temps_init = []

                metrics = logger.compute_metrics(domain_id, epoch, domain_acc, preds=preds, targets=targets, expert_usage=expert_usage)
                metrics["overall_acc"] = float(overall_acc)
                # attach routing temperature info so it is persisted in the logger
                metrics["routing_temps"] = routing_temps
                metrics["routing_temps_init"] = routing_temps_init
                logger.log(metrics)

                print(
                    f"Epoch {epoch+1} | "
                    f"Domain: {domain_acc[domain_id]:.4f} | "
                    f"Overall: {overall_acc:.4f}"
                )

                global_epoch += 1

            logger.finalize_domain(domain_acc, preds, targets)

        # end of training; save logger
        out = logger.save()
        print("Saved experiment log to:", out)

    except Exception as exc:
        # attempt to save partial log and re-raise after printing
        print("ERROR during experiment:", exc)
        try:
            if 'logger' in locals():
                err_entry = {"error": str(exc), "time": datetime.now().isoformat()}
                try:
                    logger.history.append(err_entry)
                    out = logger.save()
                    print("Saved partial logger to:", out)
                except Exception:
                    print("Failed to save logger after error")
        except Exception:
            pass
        raise
    finally:
        if temp_cfg_path is not None and os.path.exists(temp_cfg_path):
            try:
                os.remove(temp_cfg_path)
            except Exception:
                pass


if __name__ == '__main__':
    main()
