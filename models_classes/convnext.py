"""
ConvNeXt model factory and self-contained train/test helpers.
Provides a single entry function `create_convnext(...)` which returns a dictionary
containing: model, loss_fn, optimizer, train_fn, test_fn, backward_fn and metadata.

The train/test implementations are compatible with the continual-learning notebook's
expected signatures.
"""
from collections import defaultdict
import time
import numpy as np

import torch
from torch import nn
import torchvision


def train(dataloader, model, loss_fn, optimizer, test_dataloader=None, test_fn=None,
          test_interval='class', test_every_n=100, class_order=None,
          router_freeze_after_batches: int = None,
          router_balancing: bool = False, router_balance_strength: float = 0.1,
          dataset_manager=None, replay_batch_size: int = 0, replay_weight: float = 1.0):
    """Continual-stream training loop compatible with the notebook.

    This trainer now auto-detects a *shuffled* DataLoader (standard training)
    and will skip class-boundary logic in that case while still tracking
    per-class statistics. When training is shuffled, intermediate "per-class"
    tests are skipped (a single final evaluation is run if the user requested
    `test_interval=='class'`).
    """
    device = next(model.parameters()).device
    model.train()
    batch_count = 0
    current_class = None
    class_batch_counts = defaultdict(int)
    class_losses = defaultdict(float)
    training_metrics = {}
    test_history = []
    class_change_steps = []

    # Detect whether the provided dataloader is a shuffled sampler (standard
    # DataLoader with shuffle=True uses RandomSampler / SubsetRandomSampler).
    is_shuffled = False
    try:
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None:
            sname = sampler.__class__.__name__
            if sname in ('RandomSampler', 'SubsetRandomSampler'):
                is_shuffled = True
    except Exception:
        is_shuffled = False

    if is_shuffled:
        print('Shuffled dataloader detected — treating training as standard shuffled training (no per-class boundaries)')

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_class = int(y[0].item())

        # freeze router when requested (no-op if model doesn't support it)
        if router_freeze_after_batches is not None and batch_count == router_freeze_after_batches:
            if hasattr(model, 'freeze_routing'):
                try:
                    model.freeze_routing(True)
                except Exception:
                    pass

        # Only treat class-boundary events for *continual* (class-ordered) streams
        if not is_shuffled:
            if current_class != y_class:
                if current_class is not None:
                    avg_loss = class_losses[current_class] / class_batch_counts[current_class]
                    training_metrics[current_class] = {
                        'samples': class_batch_counts[current_class],
                        'avg_loss': avg_loss
                    }
                    print(f"  Class {current_class} - Processed {class_batch_counts[current_class]} samples, Avg loss: {avg_loss:>7f}")

                    if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
                        print(f"  Testing after Class {current_class}:")
                        test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                        test_result['step'] = current_class
                        test_result['step_type'] = 'class'
                        test_history.append(test_result)
                        model.train()

                current_class = y_class
                print(f"Starting training on Class {current_class}")
                class_change_steps.append(batch)
        else:
            # shuffled training: do not emit class-boundary events; just aggregate
            # per-class statistics as samples arrive
            pass

        pred = model(X)
        loss = loss_fn(pred, y)

        # optional router-balancing auxiliary loss (only affects models with MoE-like gates)
        if router_balancing:
            bal_loss = 0.0
            for m in model.modules():
                gp = getattr(m, '_last_gate_probs', None)
                ne = getattr(m, 'num_experts', None)
                if gp is not None and ne is not None:
                    p_mean = gp.mean(dim=0)  # (E,)
                    target = torch.full_like(p_mean, 1.0 / float(max(1, ne)))
                    bal_loss = bal_loss + ((p_mean - target) ** 2).mean()
            if isinstance(bal_loss, torch.Tensor):
                loss = loss + router_balance_strength * bal_loss

        # optional replay sampling
        if hasattr(model, 'replay_buffer') and replay_batch_size > 0 and model.replay_buffer.size() > 0:
            Xr_cpu, yr_cpu = model.replay_buffer.sample(replay_batch_size)
            if Xr_cpu.numel() > 0:
                Xr = Xr_cpu.to(device)
                yr = yr_cpu.to(device).long()
                pred_r = model(Xr)
                loss_r = loss_fn(pred_r, yr)
                loss = loss + replay_weight * loss_r

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # optionally let the dataset manager add samples to replay buffer
        if dataset_manager is not None and hasattr(dataset_manager, 'add_to_replay_if_present'):
            try:
                dataset_manager.add_to_replay_if_present(model, X.detach().cpu(), y.detach().cpu(), losses=None)
            except Exception:
                pass

        # accumulate per-class stats (works for both continual and shuffled streams)
        class_batch_counts[y_class] += 1
        class_losses[y_class] += loss.item()
        batch_count += 1

        if batch % 100 == 0 and batch > 0:
            print(f"  Batch {batch}: loss: {loss.item():>7f}")
            if test_interval == 'batch' and batch % test_every_n == 0 and test_dataloader is not None and test_fn is not None:
                print(f"  Testing at Batch {batch}:")
                test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                test_result['step'] = batch
                test_result['step_type'] = 'batch'
                test_history.append(test_result)
                model.train()

    # Finalize per-class training metrics for all seen classes
    for cls in sorted(class_batch_counts.keys()):
        cnt = class_batch_counts[cls]
        avg = class_losses[cls] / cnt if cnt > 0 else float('nan')
        training_metrics[cls] = {'samples': cnt, 'avg_loss': avg}

    # If this was a continual (class-ordered) stream, retain the original
    # behaviour of reporting/ testing after the last class. For shuffled
    # training we skip intermediate per-class testing — but if the user set
    # TEST_INTERVAL=='class' we run a single final evaluation so test_history
    # remains populated.
    if not is_shuffled:
        if current_class is not None:
            print(f"  Class {current_class} - Processed {class_batch_counts[current_class]} samples, Avg loss: {training_metrics[current_class]['avg_loss']:>7f}")

            if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
                print(f"  Testing after Class {current_class}:")
                test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
                test_result['step'] = current_class
                test_result['step_type'] = 'class'
                test_history.append(test_result)
                model.train()
    else:
        if test_interval == 'class' and test_dataloader is not None and test_fn is not None:
            print('Shuffled training detected — running final evaluation (intermediate per-class tests skipped).')
            test_result = test_fn(test_dataloader, model, loss_fn, class_order=class_order)
            test_result['step'] = batch_count
            test_result['step_type'] = 'shuffled'
            test_history.append(test_result)

    print(f"Training complete. Total batches: {batch_count}\n")
    return training_metrics, test_history, class_change_steps


def test(dataloader, model, loss_fn, class_order=None):
    """Evaluation loop compatible with the notebook's metrics code."""
    device = next(model.parameters()).device
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    current_class = None
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_losses = defaultdict(float)
    class_batch_counts = defaultdict(int)
    confusion_matrix = {}
    test_metrics = {}

    batch = -1
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_class = int(y[0].item())

            if current_class != y_class:
                if current_class is not None:
                    class_acc = (100 * class_correct[current_class] / class_total[current_class])
                    class_avg_loss = class_losses[current_class] / class_batch_counts[current_class]
                    test_metrics[current_class] = {
                        'accuracy': class_acc,
                        'avg_loss': class_avg_loss,
                        'total_samples': class_total[current_class]
                    }
                    print(f"  Class {current_class} - Accuracy: {class_acc:>0.1f}%, Avg loss: {class_avg_loss:>8f}")
                current_class = y_class
                print(f"Testing on Class {current_class}")

            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            total += y.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            y_class_label = int(y[0].item())
            pred_class_label = int(pred.argmax(1)[0].item())
            class_total[y_class_label] += 1
            class_losses[y_class_label] += loss.item()
            class_batch_counts[y_class_label] += 1
            class_correct[y_class_label] += (pred.argmax(1) == y).type(torch.float).sum().item()

            if y_class_label not in confusion_matrix:
                confusion_matrix[y_class_label] = defaultdict(int)
            confusion_matrix[y_class_label][pred_class_label] += 1

    # Final class stat
    per_class_accuracies = {}
    if current_class is not None:
        class_acc = (100 * class_correct[current_class] / class_total[current_class])
        class_avg_loss = class_losses[current_class] / class_batch_counts[current_class]
        test_metrics[current_class] = {
            'accuracy': class_acc,
            'avg_loss': class_avg_loss,
            'total_samples': class_total[current_class]
        }
        print(f"  Class {current_class} - Accuracy: {class_acc:>0.1f}%, Avg loss: {class_avg_loss:>8f}")

    for cls in sorted(test_metrics.keys()):
        per_class_accuracies[cls] = test_metrics[cls]['accuracy']

    avg_loss = test_loss / (batch + 1) if batch >= 0 else 0
    overall_accuracy = 100 * correct / total if total > 0 else 0
    average_task_accuracy = np.mean(list(per_class_accuracies.values())) if per_class_accuracies else 0

    test_metrics['overall'] = {
        'accuracy': overall_accuracy,
        'avg_loss': avg_loss,
        'total_samples': total,
        'average_task_accuracy': average_task_accuracy,
        'per_class_accuracies': per_class_accuracies,
        'confusion_matrix': dict(confusion_matrix)
    }

    print(f"\nTest Summary:")
    print(f"  Overall Accuracy: {overall_accuracy:>0.1f}%")
    print(f"  Average Task Accuracy: {average_task_accuracy:>0.1f}%")
    print(f"  Overall Avg Loss: {avg_loss:>8f}\n")

    return test_metrics


def backward_fn(loss, optimizer=None):
    """A minimal backward-pass helper (per-model customizations can go here)."""
    loss.backward()
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad()


def create_convnext(num_classes=1000, device=None, lr=1e-3, pretrained=False):
    """Factory that builds a convnext model and returns the trainer/tester + metadata.

    Returns a dict with keys:
      - model, loss_fn, optimizer, train_fn, test_fn, backward_fn, name, default_params
    """
    model = torchvision.models.convnext_tiny(pretrained=pretrained, num_classes=num_classes)
    if device is not None:
        model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return {
        'model': model,
        'loss_fn': loss,
        'optimizer': optimizer,
        'train_fn': train,
        'test_fn': test,
        'backward_fn': backward_fn,
        'name': 'convnext_tiny',
        'default_params': {'lr': lr, 'pretrained': pretrained, 'num_classes': num_classes}
    }
