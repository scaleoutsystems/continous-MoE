import json
import torch
import random
import numpy as np
from typing import Dict, Any
from pathlib import Path
import re

from dataset_fcns.dataset_utils import create_dataloaders
from models_fcns.model_utils import create_model

# new loss functions
from loss_classes.losses import (
    WeightedCrossEntropy,
    FocalLoss,
    LOSS_WEIGHT_UPPER_LIMIT,
)



def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    """Read a JSON experiment configuration and instantiate components.

    The returned dictionary contains the raw config under key "cfg" plus
    objects created from the configuration, e.g. dataset, loaders, model,
    optimizer, scheduler, etc.
    """
    # load JSON but allow C-style comments (// and /* */)
    with open(path) as f:
        raw = f.read()
    # remove line comments
    raw = re.sub(r"//.*", "", raw)
    # remove block comments
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    cfg = json.loads(raw)

    print(f"Loading experiment configuration from {path}")
    # always use a central results directory for outputs

    Path("results").mkdir(exist_ok=True)

    # show a compact summary of the most important fields
    epochs_per_domain = cfg.get("epochs_per_domain")

    print(f" dataset         : {cfg.get('dataset')} @ {cfg.get('dataset_root','./datasets')}")
    print(f" partitions      : {cfg.get('num_partitions')}  type={cfg.get('partition',{}).get('type')} ")
    print(f" model           : {cfg.get('model',{}).get('name')}")
    if 'patch_size' in cfg.get('model', {}):
        print(f" patch_size       : {cfg['model']['patch_size']}")
    # print additional model-specific MoE options if present
    mcfg_preview = cfg.get('model', {})
    mname = mcfg_preview.get('name', '').lower() if isinstance(mcfg_preview.get('name', ''), str) else ''
    if mname in ("pretrained_vit_moe_head", "pretrained_vit_proto_moe", "vit_moe_proto"):
        print(f" MoE experts     : {mcfg_preview.get('num_experts', 4)}")
        print(f" Prototype EMA    : {mcfg_preview.get('prototype_ema', 0.99)} | router_temp: {mcfg_preview.get('router_temperature', 0.1)}")
        print(f" Loss flags       : attraction={mcfg_preview.get('use_attraction', True)}, repulsion={mcfg_preview.get('use_repulsion', True)}, consistency={mcfg_preview.get('use_consistency', True)}, global_class={mcfg_preview.get('use_global_classification', False)}")
    print(f" epochs_per_domain  : {epochs_per_domain}")
    # full config dump for reference
    print(json.dumps(cfg, indent=2))

    # seed handling: allow multiple independent seeds; value 0 or the string
    # "random" means "do not set a seed" so the RNGs behave nondeterministically.
    seeds = cfg.get("seeds", {})

    def _norm(s):
        if s is None:
            return None
        if isinstance(s, str) and s.lower() == "random":
            return None
        try:
            iv = int(s)
            if iv == 0:
                return None
            return iv
        except Exception:
            return None

    global_seed = _norm(seeds.get("global", cfg.get("seed", None)))
    dataset_seed = _norm(seeds.get("dataset", global_seed))
    model_seed = _norm(seeds.get("model", global_seed))
    training_seed = _norm(seeds.get("training", global_seed))
    pretrain_seed = _norm(seeds.get("pretrain", global_seed))
    baseline_seed = _norm(seeds.get("baseline", global_seed))
    replay_seed = _norm(seeds.get("replay", global_seed))

    # apply global seed if requested; per-stage code (e.g. dataset loader)
    # will consider the other values separately.
    if global_seed is not None:
        _set_seed(global_seed)

    # show seed summary (None means random/unspecified)
    print(
        f" seeds: global={global_seed}, dataset={dataset_seed}, model={model_seed}, training={training_seed}, pretrain={pretrain_seed}, baseline={baseline_seed}"
    )

    # create dataset and dataloaders first; pass seeds dict so loader can
    # use the partition-specific seed if provided
    # Dataset-specific validation: for CORe50 domain-incremental splits the
    # number of chosen settings must be divisible by num_partitions so that
    # each partition (domain) receives an equal number of settings.
    if cfg.get("dataset", "").lower() == "core50":
        partition_type = cfg.get("partition", {}).get("type", "random")
        if partition_type == "domainIncremental":
            settings = cfg.get("settings", None)
            settings_len = len(settings) if settings is not None else 11
            num_parts = cfg.get("num_partitions", 1)
            if settings_len % max(1, num_parts) != 0:
                raise RuntimeError(
                    f"For domainIncremental partitioning, number of settings ({settings_len}) must be divisible by num_partitions ({num_parts})"
                )

    data_objs = create_dataloaders(cfg)
    dataset = data_objs["dataset"]
    train_loaders = data_objs["train_loaders"]
    test_loaders = data_objs["test_loaders"]
    pretrain_loader = data_objs.get("pretrain_loader", None)
    partition_distributions = data_objs.get("partition_distributions", None)

    # determine image size from the dataset if possible and store in the
    # model config so factories can adjust architectures automatically.
    # we peek at the first training batch (or fall back to dataset[0]).
    img_size = None
    if train_loaders and len(train_loaders) > 0:
        try:
            for imgs, _ in train_loaders[0]:
                if imgs is not None and imgs.ndim >= 4:
                    img_size = imgs.shape[2]
                break
        except Exception:
            img_size = None
    if img_size is None and dataset is not None:
        try:
            sample = dataset[0][0]
            if sample.ndim >= 3:
                img_size = sample.shape[1]
        except Exception:
            img_size = None
    if img_size is not None:
        mcfg = cfg.setdefault("model", {})
        if mcfg.get("img_size") is None:
            mcfg["img_size"] = img_size

    # build model
    if model_seed is not None:
        torch.manual_seed(model_seed)
        
    model = create_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # optimizer
    opt_cfg = cfg.get("optimizer", {"name": "adam"})
    loss_cfg = cfg.get("loss", {})
    lr = loss_cfg.get("lr", 1e-3)

    # MoE expert LR multipliers (optional). These may be provided in the
    # model config as either a scalar or a list. If omitted, experts use the
    # base learning rate.
    model_cfg = cfg.get("model", {})
    _um = model_cfg.get("moe_unshared_lr_multipliers", None)
    _sm = model_cfg.get("moe_shared_lr_multipliers", None)
    _ss = model_cfg.get("moe_shared_lr_multiplier", None)

    def _normalize_mult(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            return [float(x) for x in v]
        return None

    unshared_multipliers = _normalize_mult(_um)
    shared_multipliers = _normalize_mult(_sm)
    shared_scalar = None if _ss is None else float(_ss)

    # optionally reduce router learning rate
    router_mult = cfg.get("router_lr_multiplier", 1.0)

    # collect router parameters (if supported)
    router_params = list(model.get_router_parameters()) if hasattr(model, "get_router_parameters") else []
    router_ids = {id(p) for p in router_params}

    # if router_mult <= 0 treat it as freeze: disable grads and exclude them
    if router_mult is None:
        router_mult = 1.0
    if router_mult <= 0 and hasattr(model, "freeze_routing"):
        model.freeze_routing(True)
        router_params = []
        router_ids = set()

    # collect expert-specific parameter groups (if any MoE modules present)
    expert_param_groups = []
    expert_param_ids = set()
    for m in model.modules():
        if hasattr(m, "experts") and isinstance(getattr(m, "experts"), torch.nn.ModuleList) and hasattr(m, "num_unshared_experts"):
            if hasattr(m, "get_expert_parameters"):
                expert_infos = m.get_expert_parameters()
            else:
                expert_infos = []
                for idx, expert in enumerate(m.experts):
                    expert_infos.append((idx, list(expert.parameters()), idx >= m.num_unshared_experts))

            for e_idx, params, is_shared in expert_infos:
                if not params:
                    continue
                # determine multiplier for this expert
                mult = 1.0
                if not is_shared:
                    if unshared_multipliers is None:
                        mult = 1.0
                    elif isinstance(unshared_multipliers, float):
                        mult = float(unshared_multipliers)
                    elif isinstance(unshared_multipliers, list):
                        if e_idx < len(unshared_multipliers):
                            mult = float(unshared_multipliers[e_idx])
                        else:
                            mult = float(unshared_multipliers[-1])
                else:
                    if shared_multipliers is not None:
                        if isinstance(shared_multipliers, list):
                            sidx = e_idx - m.num_unshared_experts
                            if sidx < len(shared_multipliers):
                                mult = float(shared_multipliers[sidx])
                            else:
                                mult = float(shared_multipliers[-1])
                        else:
                            mult = float(shared_multipliers)
                    elif shared_scalar is not None:
                        mult = float(shared_scalar)
                    else:
                        mult = 1.0

                # skip identity multiplier
                if mult == 1.0:
                    continue

                # filter out params already handled (router or other experts)
                filtered = [p for p in params if id(p) not in router_ids and id(p) not in expert_param_ids]
                if not filtered:
                    continue
                for p in filtered:
                    expert_param_ids.add(id(p))
                expert_param_groups.append({"params": filtered, "lr": lr * float(mult)})

    # base parameters: exclude router and expert params
    all_params = list(model.parameters())
    base_params = [p for p in all_params if id(p) not in router_ids and id(p) not in expert_param_ids]

    param_groups = [{"params": base_params}]
    if router_params:
        param_groups.append({"params": router_params, "lr": lr * router_mult})
    # append any expert groups discovered
    param_groups.extend(expert_param_groups)

    # construct optimizer
    name = opt_cfg["name"].lower()
    if name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    elif name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=opt_cfg.get("momentum", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer {opt_cfg['name']}")

    # scheduler helpers
    def _make_scheduler(optimizer, sch_cfg, epochs_per_domain):
        if not sch_cfg or sch_cfg.get("name") is None:
            return None

        name = sch_cfg.get("name")
        if name is None:
            return None

        name = name.lower()
        if name == "cosine":
            T_max = sch_cfg.get("T_max", epochs_per_domain)
            if T_max is None or T_max == 0:
                T_max = epochs_per_domain
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        if name == "step":
            step_size = sch_cfg.get("step_size", 10)
            gamma = sch_cfg.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if name == "linear":
            start_factor = sch_cfg.get("start_factor", 0.1)
            end_factor = sch_cfg.get("end_factor", 1.0)
            total_iters = sch_cfg.get("total_iters", epochs_per_domain)
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
            )

        print(f"Warning: unknown scheduler {sch_cfg.get('name')} -- skipping")
        return None

    scheduler = _make_scheduler(optimizer, cfg.get("scheduler", {}), epochs_per_domain)
    pretrain_scheduler = _make_scheduler(
        optimizer,
        cfg.get("pretrain_scheduler", {}),
        epochs_per_domain,
    )

    # loss
    loss_cfg = cfg.get("loss", {"name": "cross_entropy"})
    loss_name = loss_cfg.get("name", "cross_entropy").lower()
    # common weighting options
    weighted = loss_cfg.get("weighted", False)
    upper_lim = loss_cfg.get("weight_upper_limit", LOSS_WEIGHT_UPPER_LIMIT)
    cumul = loss_cfg.get("weight_cumulative", False)

    if loss_name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    elif loss_name == "weighted_cross_entropy":
        num_classes = cfg.get("model", {}).get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes must be specified in config to use weighted loss")
        criterion = WeightedCrossEntropy(num_classes, upper_limit=upper_lim)
        criterion._weight_cumulative = cumul

    elif loss_name == "focal":
        num_classes = cfg.get("model", {}).get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes must be specified in config to use focal loss")
        alpha = loss_cfg.get("alpha", 0.25)
        gamma = loss_cfg.get("gamma", 2.0)
        criterion = FocalLoss(
            num_classes,
            alpha=alpha,
            gamma=gamma,
            weight_upper_limit=upper_lim,
            weighted=weighted,
        )
        criterion._weight_cumulative = cumul

    else:
        raise ValueError(f"Unsupported loss {loss_name}")

    # ensure the loss module lives on the correct device
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    return {
        "cfg": cfg,
        "dataset": dataset,
        "train_loaders": train_loaders,
        "test_loaders": test_loaders,
        "pretrain_loader": pretrain_loader,
        "partition_distributions": partition_distributions,
        "train_frac": data_objs.get("train_frac"),
        "batch_size": data_objs.get("batch_size"),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "pretrain_scheduler": pretrain_scheduler,
        "criterion": criterion,
        "device": device,
        # resolved seeds (None means random/unspecified)
        "resolved_seeds": {
            "global": global_seed,
            "dataset": dataset_seed,
            "model": model_seed,
            "training": training_seed,
            "pretrain": pretrain_seed,
            "baseline": baseline_seed,
            "replay": replay_seed,
        },
    }
