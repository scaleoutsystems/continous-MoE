import torch
import matplotlib.pyplot as plt
from typing import Tuple


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, active_params).

    ``total_params`` is a straightforward sum over all model parameters.
    ``active_params`` attempts to approximate the number of parameters that
    would be touched during a forward pass under the MoE routing assumptions.

    In particular, non-MoE modules contribute all of their parameters.  For an
    MoE layer we assume that the router will execute at most ``top_k``
    *unshared* experts plus *all* shared experts, and we always include the
    router's own parameters.  Thus the worst-case cost for a layer is

        expert_size * (num_shared_experts + top_k) + router_params

    where ``expert_size`` is taken from a single expert module.  This yields a
    tighter and more accurate estimate than the previous simplistic method.
    """
    parameters = list(model.parameters())
    unique = {p.data_ptr(): p for p in parameters}.values()
    total = sum(p.numel() for p in unique)
    active = total

    # iterate through modules and accumulate inactive reduction.  when an MoE
    # module is encountered we compute its layer reduction explicitly and
    # skip over its submodules to avoid double-counting.
    for m in model.modules():
        from models_classes.moe_vit import MoE  # avoid circular import at module load
        if isinstance(m, MoE):
            # size of a single expert
            if len(m.experts) > 0:
                expert_params = int(sum(p.numel() for p in m.experts[0].parameters()))
            else:
                expert_params = 0
            # Remove number of unused unshared experts in each forward pass at this layer
            layer_inactive = expert_params * (m.num_unshared_experts - m.top_k)
            active -= layer_inactive
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
