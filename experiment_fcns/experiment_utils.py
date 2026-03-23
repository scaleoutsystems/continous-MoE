import torch


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
