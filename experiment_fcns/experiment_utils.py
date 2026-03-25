import torch

# auxiliary loss (router balancing, etc.)
def compute_aux_loss(model, cfg):
    loss_aux = 0.0
    if cfg.get("router_balancing", False):
        strength = cfg.get("router_balance_strength", 0.1)
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
    return loss_aux

def _make_optimizer_for_model(m, cfg):
        opt_cfg = cfg.get("optimizer", {"name": "adam"})
        lr = cfg.get("loss", {}).get("lr", opt_cfg.get("lr", 1e-3))
        if opt_cfg["name"].lower() == "adam":
            return torch.optim.Adam(m.parameters(), lr=lr)
        elif opt_cfg["name"].lower() == "sgd":
            return torch.optim.SGD(m.parameters(), lr=lr, momentum=opt_cfg.get("momentum", 0.0))
        elif opt_cfg["name"].lower() == "adamw":
            return torch.optim.AdamW(m.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer {opt_cfg["name"]}")

def _make_scheduler_for_optimizer(opt, cfg, epochs_per_domain):
    sch_cfg = cfg.get("scheduler", {})
    if not sch_cfg or sch_cfg.get("name") is None:
        return None
    name = sch_cfg.get("name").lower()
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
