import json
import torch
import random
import numpy as np
from typing import Dict, Any
from pathlib import Path
import re

from dataset_fcns.dataset_utils import create_dataloaders
from models_fcns.model_utils import create_model


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
    replay_seed = _norm(seeds.get("replay", global_seed))

    # apply global seed if requested; per-stage code (e.g. dataset loader)
    # will consider the other values separately.
    if global_seed is not None:
        _set_seed(global_seed)

    # show seed summary (None means random/unspecified)
    print(
        f" seeds: global={global_seed}, dataset={dataset_seed}, model={model_seed}, training={training_seed}"
    )

    # create dataset and dataloaders first; pass seeds dict so loader can
    # use the partition-specific seed if provided
    data_objs = create_dataloaders(cfg)
    dataset = data_objs["dataset"]
    train_loaders = data_objs["train_loaders"]
    test_loaders = data_objs["test_loaders"]
    pretrain_loader = data_objs.get("pretrain_loader", None)

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
    # optionally reduce router learning rate
    router_mult = cfg.get("router_lr_multiplier", 1.0)
    if opt_cfg["name"].lower() == "adam":
        # if multiplier is zero we treat it as a freeze
        if router_mult is None:
            router_mult = 1.0
        if router_mult <= 0 or not hasattr(model, "get_router_parameters"):
            if router_mult <= 0 and hasattr(model, "freeze_routing"):
                model.freeze_routing(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            router_params = list(model.get_router_parameters())
            non_router = [p for p in model.parameters() if p not in set(router_params)]
            param_groups = [{"params": non_router}]
            if router_params:
                param_groups.append({"params": router_params, "lr": lr * router_mult})
            optimizer = torch.optim.Adam(param_groups, lr=lr)
    elif opt_cfg["name"].lower() == "adamw":
        # if multiplier is zero we treat it as a freeze
        if router_mult is None:
            router_mult = 1.0
        if router_mult <= 0 or not hasattr(model, "get_router_parameters"):
            if router_mult <= 0 and hasattr(model, "freeze_routing"):
                model.freeze_routing(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            router_params = list(model.get_router_parameters())
            non_router = [p for p in model.parameters() if p not in set(router_params)]
            param_groups = [{"params": non_router}]
            if router_params:
                param_groups.append({"params": router_params, "lr": lr * router_mult})
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        
    elif opt_cfg["name"].lower() == "sgd":
        if router_mult is None:
            router_mult = 1.0
        if router_mult <= 0 or not hasattr(model, "get_router_parameters"):
            if router_mult <= 0 and hasattr(model, "freeze_routing"):
                model.freeze_routing(True)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=opt_cfg.get("momentum", 0.0))
        else:
            router_params = list(model.get_router_parameters())
            non_router = [p for p in model.parameters() if p not in set(router_params)]
            param_groups = [{"params": non_router}]
            if router_params:
                param_groups.append({"params": router_params, "lr": lr * router_mult})
            optimizer = torch.optim.SGD(param_groups, lr=lr,
                                        momentum=opt_cfg.get("momentum", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer {opt_cfg['name']}")

    # scheduler
    sch_cfg = cfg.get("scheduler", {})
    scheduler = None
    if sch_cfg:
        if sch_cfg.get("name") == "cosine":
            T_max = epochs_per_domain if sch_cfg.get("T_max", None) is (None or 0) else sch_cfg.get("T_max")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max)
        elif sch_cfg.get("name") == "none":
            scheduler = None
        else:
            print(f"Warning: unknown scheduler {sch_cfg.get('name')} -- skipping")

    # loss
    loss_cfg = cfg.get("loss", {"name": "cross_entropy"})
    if loss_cfg["name"] == "cross_entropy" or loss_cfg["name"] == "weighted_cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss {loss_cfg['name']}")

    return {
        "cfg": cfg,
        "dataset": dataset,
        "train_loaders": train_loaders,
        "test_loaders": test_loaders,
        "pretrain_loader": pretrain_loader,
        "train_frac": data_objs.get("train_frac"),
        "batch_size": data_objs.get("batch_size"),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "device": device,
    }
