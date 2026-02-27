import torch
import matplotlib.pyplot as plt
from typing import Tuple


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, active_params).

    For standard architectures both numbers are the same.  For an MoE model we
    estimate "active" parameters by looking at the router configuration and
    assuming that only top-k experts are executed for each sample; this is a
    rough upper bound (k * expert_size) but gives an idea of sparsity.
    """
    total = int(sum(p.numel() for p in model.parameters()))
    active = total
    # heuristic for MoE
    if hasattr(model, 'get_moe_utilization'):
        # assume worst‑case k experts active per layer
        act = 0
        for m in model.modules():
            if hasattr(m, 'num_experts') and hasattr(m, 'top_k'):
                # each expert is an MLP: estimate parameter count as total/num_experts
                # type: ignore - runtime check ensures ModuleList exists
                expert_params = int(sum(p.numel() for p in m.experts[0].parameters()))  # type: ignore
                act += expert_params * int(getattr(m, 'top_k', 0))
            else:
                act += int(sum(p.numel() for p in m.parameters()))
        active = act
    return total, int(active)


def plot_expert_loads(model: torch.nn.Module):
    """Draw a figure summarizing expert utilization for MoE layers.

    The model must implement `get_moe_utilization()` and return a list of
    dicts with keys 'layer_index', 'fraction', and 'samples'.
    """
    if not hasattr(model, 'get_moe_utilization'):
        print("Model has no MoE layers; skipping expert load plot")
        return
    stats = model.get_moe_utilization()  # type: ignore
    if len(stats) == 0:
        print("No MoE layers found or no statistics recorded yet")
        return
    num_layers = len(stats)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 3*num_layers))
    if num_layers == 1:
        axes = [axes]
    for ax, s in zip(axes, stats):
        ax.bar(range(len(s['fraction'])), s['fraction'])
        ax.set_title(f"Layer {s['layer_index']} expert fractions ({s['samples']} samples)")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Mean probability")
    plt.tight_layout()
    plt.show()
