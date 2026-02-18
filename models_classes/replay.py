"""Simple replay buffer for continual learning experiments.

Provides a small, dependency-free buffer with FIFO and reservoir sampling
policies plus basic sample/add APIs used by the notebook and models.
"""
from typing import List, Tuple, Optional
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 1000, policy: str = 'fifo'):
        assert capacity > 0
        assert policy in ('fifo', 'random')
        self.capacity = int(capacity)
        self.policy = policy
        self._X: List[torch.Tensor] = []
        self._y: List[torch.Tensor] = []
        self._n_seen = 0  # for reservoir sampling

    def add_batch(self, X_cpu: torch.Tensor, y_cpu: torch.Tensor, losses_cpu: Optional[torch.Tensor] = None):
        """Add a batch of CPU tensors to the buffer.
        X_cpu: Tensor [B, ...] on CPU
        y_cpu: Tensor [B]
        losses_cpu: optional Tensor [B] used by advanced policies (not used here)
        """
        B = X_cpu.size(0)
        for i in range(B):
            xi = X_cpu[i].clone()
            yi = y_cpu[i].clone()
            self._n_seen += 1
            if len(self._X) < self.capacity:
                self._X.append(xi)
                self._y.append(yi)
            else:
                if self.policy == 'fifo':
                    # simple overwrite oldest
                    self._X.pop(0)
                    self._y.pop(0)
                    self._X.append(xi)
                    self._y.append(yi)
                else:
                    # reservoir sampling: replace an existing item with prob capacity/n_seen
                    r = random.randint(0, self._n_seen - 1)
                    if r < self.capacity:
                        self._X[r] = xi
                        self._y[r] = yi

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items (without replacement). Returns CPU tensors."""
        n = min(int(n), len(self._X))
        if n == 0:
            return torch.empty(0), torch.empty(0)
        idxs = random.sample(range(len(self._X)), n)
        Xs = torch.stack([self._X[i] for i in idxs], dim=0)
        ys = torch.stack([self._y[i] for i in idxs], dim=0)
        return Xs, ys

    def size(self) -> int:
        return len(self._X)

    def clear(self):
        self._X.clear()
        self._y.clear()
        self._n_seen = 0
