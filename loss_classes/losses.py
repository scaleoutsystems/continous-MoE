import torch
import torch.nn as nn
import torch.nn.functional as F

# upper bound to clamp class weights
LOSS_WEIGHT_UPPER_LIMIT = 10.0


def _compute_weights(counts: torch.Tensor, upper_limit: float):
    """Turn a 1D tensor of class counts into inverse-frequency weights.

    The returned weights are clamped to ``upper_limit`` and will never be
    NaN (missing classes are handled gracefully by clamping).
    """
    counts = counts.float()
    total = counts.sum()
    if total <= 0:
        # no samples at all - fall back to uniform
        return torch.ones_like(counts)
    # add small epsilon to avoid divide-by-zero
    weights = total / (counts + 1e-6)
    if upper_limit is not None:
        weights = torch.clamp(weights, max=upper_limit)
    return weights


class WeightedCrossEntropy(nn.Module):
    """Cross-entropy loss where each class has a data-dependent weight.

    Weights can be updated via :meth:`update_weights`; they default to
    uniform ones.  The class also supports a ``cumulative`` mode in which the
    new weight vector is averaged with the previous weights.  This matches the
    behaviour requested by the user description.
    """

    def __init__(self, num_classes: int, upper_limit: float = LOSS_WEIGHT_UPPER_LIMIT):
        super().__init__()
        self.num_classes = num_classes
        self.upper_limit = upper_limit
        # store weights as a buffer so they move with the module
        self.register_buffer('weights', torch.ones(num_classes))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(input, target, weight=self.weights, reduction='mean')

    def update_weights(self, counts, cumulative: bool = False):
        """Recompute weights from a vector of class counts.

        Args:
            counts: iterable or tensor giving number of samples for each class
            cumulative: if True, average the newly computed weights with the
                existing weights instead of replacing them outright.
        """
        counts_tensor = torch.as_tensor(counts, dtype=torch.float, device=self.weights.device)
        new_w = _compute_weights(counts_tensor, self.upper_limit)
        if cumulative:
            self.weights = 0.5 * (self.weights + new_w)
        else:
            self.weights = new_w


class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin *et al.*) with optional class weighting.

    ``alpha`` may be a scalar or a tensor of per-class factors.  If it is
    ``None`` or the string ``"auto"`` the alpha weights are computed
    automatically from class frequencies each time :meth:`update_weights` is
    called (same behaviour as ``WeightedCrossEntropy``).  This makes the loss
    adapt to imbalanced partitions.

    The implementation is compatible with the update_weights interface used by
    :class:`WeightedCrossEntropy`.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float | list | torch.Tensor | None = 0.25,
        gamma: float = 2.0,
        weight_upper_limit: float = LOSS_WEIGHT_UPPER_LIMIT,
        weighted: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.weighted = weighted
        self.upper_limit = weight_upper_limit
        # determine whether alpha should be auto-updated like weights
        self.auto_alpha = False
        if alpha is None or (isinstance(alpha, str) and alpha.lower() == "auto"):
            # use inverse-frequency weights, updated via update_weights
            self.auto_alpha = True
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            a_tensor = torch.as_tensor(alpha, dtype=torch.float)
            if a_tensor.numel() == 1:
                self.alpha = a_tensor.item()
            else:
                if a_tensor.numel() != num_classes:
                    raise ValueError(
                        f"alpha list must have length num_classes ({num_classes}), got {a_tensor.numel()}"
                    )
                self.alpha = a_tensor
        else:
            # assume numeric scalar
            self.alpha = float(alpha)
        if weighted:
            self.register_buffer('weights', torch.ones(num_classes))
        else:
            # placeholder - not used but easier for code path
            self.register_buffer('weights', torch.ones(num_classes))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # compute cross entropy per-sample; include class weights only if
        # requested.  the weight tensor is kept updated regardless, so it can
        # be inspected even when not applied.
        ce = F.cross_entropy(input, target, weight=self.weights, reduction='none')
        pt = torch.exp(-ce)  # pt = probability of the true class
        alpha_factor = self.alpha
        # if per-class alpha tensor, gather values for each target
        if isinstance(self.alpha, torch.Tensor) and self.alpha.numel() == self.num_classes:
            alpha_factor = self.alpha.to(input.device)[target]
        else:
            alpha_factor = torch.tensor(self.alpha, device=input.device) if isinstance(self.alpha, (float, int)) else self.alpha
        loss = alpha_factor * ((1 - pt) ** self.gamma) * ce
        return loss.mean()

    def update_weights(self, counts, cumulative: bool = False):
        """Update the internal weight tensor (and alpha if auto)."""
        counts_tensor = torch.as_tensor(counts, dtype=torch.float, device=self.weights.device)
        new_w = _compute_weights(counts_tensor, self.upper_limit)
        # always update stored weights so they reflect partition distribution
        if cumulative:
            self.weights = 0.5 * (self.weights + new_w)
        else:
            self.weights = new_w
        # if alpha is set to auto, mirror the same update behavior
        if self.auto_alpha:
            if cumulative and isinstance(self.alpha, torch.Tensor):
                self.alpha = 0.5 * (self.alpha + new_w)
            else:
                # alpha may be scalar or tensor, ensure tensor
                self.alpha = new_w.clone()
