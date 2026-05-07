import torch
import math


# auxiliary loss (router balancing, etc.)
def compute_aux_loss(model, cfg, epoch: int | None = None):
    """Compute auxiliary router-related losses.

    Supports annealing the router-balance strength between a max and min
    value using a cosine schedule over a configured number of epochs.

    Args:
        model: the model instance
        cfg: full experiment config dict
        epoch: optional current epoch (used for annealing)
    """
    loss_aux = 0.0
    if cfg.get("router_balancing", False):
        # read max (backwards-compatible fallback to router_balance_strength)
        max_s = float(cfg.get("router_balance_max", cfg.get("router_balance_strength", 0.1)))
        min_s = float(cfg.get("router_balance_min", 0.0))
        anneal_epochs = int(cfg.get("router_balance_anneal_epochs", 0))

        if anneal_epochs > 0 and epoch is not None:
            frac = float(max(0.0, min(1.0, float(epoch) / float(anneal_epochs))))
            # cosine anneal from max -> min over anneal_epochs
            strength = min_s + 0.5 * (max_s - min_s) * (1.0 + math.cos(math.pi * frac))
        else:
            strength = max_s

        # use helper on model if available
        if hasattr(model, "router_balance_loss"):
            loss_aux = model.router_balance_loss(strength)
        else:
            for m in model.modules():
                if hasattr(m, '_last_gate_probs') and m._last_gate_probs is not None:
                    p_mean = m._last_gate_probs.mean(dim=0)
                    target = torch.full_like(p_mean, 1.0 / float(max(1, p_mean.numel())))
                    loss_aux = loss_aux + ((p_mean - target) ** 2).mean()
            loss_aux = strength * loss_aux
    # allow models to compute richer router-related auxiliary losses (attraction,
    # repulsion, consistency) via a helper method. The model is expected to
    # cache information from the most recent forward pass.
    if hasattr(model, "router_aux_loss"):
        try:
            aux_loss_model = model.router_aux_loss(cfg.get("model", {}))
        except Exception:
            aux_loss_model = model.router_aux_loss()
        if isinstance(aux_loss_model, torch.Tensor):
            loss_aux = loss_aux + aux_loss_model
    return loss_aux


def _make_optimizer_for_model(m, cfg):
    opt_cfg = cfg.get("optimizer", {"name": "adam"})
    lr = cfg.get("loss", {}).get("lr", opt_cfg.get("lr", 1e-3))

    # support MoE per-expert multipliers if provided in cfg['model']
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

    # collect router params (if present) and optionally freeze if requested
    router_mult = cfg.get("router_lr_multiplier", 1.0)
    router_params = list(m.get_router_parameters()) if hasattr(m, "get_router_parameters") else []
    router_ids = {id(p) for p in router_params}
    if router_mult is None:
        router_mult = 1.0
    if router_mult <= 0 and hasattr(m, "freeze_routing"):
        m.freeze_routing(True)
        router_params = []
        router_ids = set()

    # collect all expert parameter ids (so we can always create a dedicated
    # expert param-group even if no multipliers are specified)
    all_expert_param_ids = set()
    for mod in m.modules():
        if hasattr(mod, "experts") and isinstance(getattr(mod, "experts"), torch.nn.ModuleList):
            for expert in mod.experts:
                for p in expert.parameters():
                    if id(p) not in router_ids:
                        all_expert_param_ids.add(id(p))

    # collect expert groups (per-expert multipliers) as before; track ids used
    expert_param_groups = []
    expert_param_ids = set()
    for mod in m.modules():
        if hasattr(mod, "experts") and isinstance(getattr(mod, "experts"), torch.nn.ModuleList):
            if hasattr(mod, "get_expert_parameters"):
                expert_infos = mod.get_expert_parameters()
            else:
                expert_infos = []
                num_unshared = getattr(mod, "num_unshared_experts", len(mod.experts))
                for idx, expert in enumerate(mod.experts):
                    expert_infos.append((idx, list(expert.parameters()), idx >= num_unshared))

            for e_idx, params, is_shared in expert_infos:
                if not params:
                    continue
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
                            sidx = e_idx - mod.num_unshared_experts
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

                if mult == 1.0:
                    continue
                filtered = [p for p in params if id(p) not in router_ids and id(p) not in expert_param_ids]
                if not filtered:
                    continue
                for p in filtered:
                    expert_param_ids.add(id(p))
                expert_param_groups.append({"params": filtered, "lr": lr * float(mult)})

    all_params = list(m.parameters())
    # base parameters = all params excluding routers and all expert params
    base_params = [p for p in all_params if id(p) not in router_ids and id(p) not in all_expert_param_ids]
    param_groups = [{"params": base_params}]
    if router_params:
        param_groups.append({"params": router_params, "lr": lr * router_mult})
    # add any per-expert groups (from multipliers)
    param_groups.extend(expert_param_groups)

    # remaining expert params not covered by specialized per-expert groups
    remaining_expert_params = [p for p in all_params if id(p) in all_expert_param_ids and id(p) not in expert_param_ids and id(p) not in router_ids]
    if remaining_expert_params:
        # allow absolute LR or multiplier in model cfg
        model_cfg = cfg.get("model", {})
        moe_expert_lr = model_cfg.get("moe_expert_lr", None)
        moe_expert_lr_mult = float(model_cfg.get("moe_expert_lr_multiplier", 1.0))
        if moe_expert_lr is not None:
            group_lr = float(moe_expert_lr)
        else:
            group_lr = lr * moe_expert_lr_mult
        param_groups.append({"params": remaining_expert_params, "lr": group_lr})

    name = opt_cfg["name"].lower()
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=opt_cfg.get("momentum", 0.0))
    elif name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer {opt_cfg['name']}")


def _make_scheduler_for_optimizer(opt, cfg, epochs_per_domain):
    sch_cfg = cfg.get("scheduler", {})
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
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=sch_cfg.get("step_size", 10), gamma=sch_cfg.get("gamma", 0.1))
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(opt, start_factor=sch_cfg.get("start_factor", 1.0), end_factor=sch_cfg.get("end_factor", 0.1), total_iters=sch_cfg.get("total_iters", epochs_per_domain))
    return None


def _norm_seed(val):
    if val is None:
        return None
    if isinstance(val, str) and val.lower() == "random":
        return None
    try:
        iv = int(val)
        if iv == 0:
            return None
        return iv
    except Exception:
        return None


def _maybe_update_router(cfg, model, optimizer, scheduler, epoch, router_frozen):
    rf = cfg.get("router_freeze_after_epochs", cfg.get("router_freeze_after_batches"))
    if rf is not None and epoch >= rf and not router_frozen and hasattr(model, "freeze_routing"):
        model.freeze_routing(True)
        router_frozen = True
        print(f"Router parameters frozen at epoch {epoch}")

    rmult_event = cfg.get("router_lr_multiplier_after_epochs", cfg.get("router_lr_multiplier_after_batches"))
    if rmult_event is not None and epoch == rmult_event:
        mult = cfg.get("router_lr_multiplier", 1.0)
        if hasattr(model, "adjust_router_learning_rate"):
            optimizer = model.adjust_router_learning_rate(optimizer, mult)
            print(f"Router lr multiplier applied: {mult} at epoch {epoch}")
            if scheduler is not None:
                scheduler.optimizer = optimizer
                scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]
    # allow modules (e.g., ImageMoE layers) to update their annealed routing
    # temperature each epoch if they expose the helper.
    for mod in model.modules():
        if hasattr(mod, "update_routing_temperature"):
            try:
                mod.update_routing_temperature(epoch)
            except Exception:
                pass

    return optimizer, router_frozen


@torch.no_grad()
def evaluate_full_test(model, test_loaders, device, num_classes=None):
    model.eval()
    all_preds = []
    all_targets = []
    for loader in test_loaders:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    if not all_preds:
        n = num_classes or 0
        return 0.0, torch.zeros((n, n), dtype=torch.long)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    overall_acc = (all_preds == all_targets).float().mean().item()
    if num_classes is None:
        num_classes = int(max(all_targets.max().item(), all_preds.max().item()) + 1)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        conf[t.long(), p.long()] += 1
    return overall_acc, conf


def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0
    return correct / total
