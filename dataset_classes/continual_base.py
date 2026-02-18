"""Shared continual-dataset helpers and base manager.

Provides:
 - group_indices_by_class(dataset)
 - ContinualClassStream (iterable dataset yielding per-class blocks)
 - ContinualDataset (manager that exposes train/test datasets and continual dataloaders)
"""
from collections import defaultdict
import torch
from torch.utils.data import DataLoader


def group_indices_by_class(dataset):
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        class_to_indices[int(y)].append(idx)
    return class_to_indices


class ContinualClassStream(torch.utils.data.IterableDataset):
    def __init__(self, dataset, class_order, class_to_indices):
        self.dataset = dataset
        self.class_order = class_order
        self.class_to_indices = class_to_indices

    def __iter__(self):
        for cls in self.class_order:
            for idx in self.class_to_indices[cls]:
                yield self.dataset[idx]


class ContinualDataset:
    """Manager for datasets used in the continual-learning notebook.

    Wraps a standard torchvision dataset pair (train / test) and exposes:
      - train_dataset, test_dataset
      - train_dataloader, test_dataloader (continual stream style)
      - class_to_indices_train/test, class_order_train/test
      - convenience helpers: get_pretrain_dataloader(), get_shuffled_dataloader(), add_to_replay_if_present()
      - name and to_dict() for metadata

    Note: several helper APIs are intentionally lightweight so that dataset wrappers
    (e.g. CIFAR10SmallContinual) can forward optional arguments like
    `pretrain_samples_per_class` without failing if the wrapper doesn't supply a
    specialized pretrain split.
    """
    def __init__(self, name, train_dataset, test_dataset, batch_size=1, num_workers=0,
                 pretrain_samples_per_class: int = 0, pretrain_reuse: bool = False,
                 shuffle_within_class: bool = False, seed: int = 0):
        self.name = name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        # optional pretrain configuration (may be 0 to disable)
        self.pretrain_samples_per_class = int(pretrain_samples_per_class or 0)
        self.pretrain_reuse = bool(pretrain_reuse)
        self.shuffle_within_class = bool(shuffle_within_class)
        self.seed = seed

        # index bookkeeping
        self.class_to_indices_train = group_indices_by_class(self.train_dataset)
        self.class_to_indices_test = group_indices_by_class(self.test_dataset)

        self.class_order_train = sorted(self.class_to_indices_train.keys())
        self.class_order_test = sorted(self.class_to_indices_test.keys())

        # continual streams (class-ordered, deterministic)
        self.train_stream = ContinualClassStream(self.train_dataset, self.class_order_train, self.class_to_indices_train)
        self.test_stream = ContinualClassStream(self.test_dataset, self.class_order_test, self.class_to_indices_test)

        self.train_dataloader = DataLoader(self.train_stream, batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_stream, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader

    # --- Convenience helpers used by the notebook ---------------------------
    def get_pretrain_dataloader(self):
        """Return a DataLoader for a small balanced pretrain split, or None.

        By default this returns None when `pretrain_samples_per_class` is 0.
        If enabled, it builds a Subset over `train_dataset` with the first
        `pretrain_samples_per_class` indices for each class (deterministic).
        """
        if self.pretrain_samples_per_class <= 0:
            return None
        from torch.utils.data import Subset
        selected = []
        for cls in self.class_order_train:
            inds = self.class_to_indices_train.get(cls, [])
            selected.extend(inds[:self.pretrain_samples_per_class])
        if not selected:
            return None
        subset = Subset(self.train_dataset, selected)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_shuffled_dataloader(self, model=None):
        """Return a conventional shuffled DataLoader for sanity-check training.
        If a model is provided, set the generator device to match the model's device (cuda or cpu).
        """
        try:
            from torch.utils.data import DataLoader as _DL
            import torch
            generator = None
            if model is not None:
                device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
                gen_device = 'cuda' if device.type == 'cuda' else 'cpu'
                generator = torch.Generator(device=gen_device)
            return _DL(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, generator=generator)
        except Exception:
            return None

    def add_to_replay_if_present(self, model, X_cpu, y_cpu, losses=None):
        """Default replay hook: add incoming batch to `model.replay_buffer` when present.

        This simple behavior (FIFO / reservoir are handled by the buffer) makes it
        possible to enable `ENABLE_REPLAY` from the notebook without modifying
        every dataset wrapper.
        """
        try:
            if hasattr(model, 'replay_buffer') and getattr(model, 'replay_buffer') is not None:
                model.replay_buffer.add_batch(X_cpu, y_cpu, losses)
                return True
        except Exception:
            pass
        return False

    # ---------------------------------------------------------------------
    def to_dict(self):
        return {
            'name': self.name,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'num_classes': len(self.class_order_train),
            'class_order_train': self.class_order_train,
            'class_order_test': self.class_order_test,
            'train_transform': repr(getattr(self.train_dataset, 'transform', None)),
            'test_transform': repr(getattr(self.test_dataset, 'transform', None)),
            'pretrain': {'samples_per_class': self.pretrain_samples_per_class, 'reuse': self.pretrain_reuse}
        }
