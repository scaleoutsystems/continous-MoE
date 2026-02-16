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
      - name and to_dict() for metadata
    """
    def __init__(self, name, train_dataset, test_dataset, batch_size=1, num_workers=0):
        self.name = name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.class_to_indices_train = group_indices_by_class(self.train_dataset)
        self.class_to_indices_test = group_indices_by_class(self.test_dataset)

        self.class_order_train = sorted(self.class_to_indices_train.keys())
        self.class_order_test = sorted(self.class_to_indices_test.keys())

        self.train_stream = ContinualClassStream(self.train_dataset, self.class_order_train, self.class_to_indices_train)
        self.test_stream = ContinualClassStream(self.test_dataset, self.class_order_test, self.class_to_indices_test)

        self.train_dataloader = DataLoader(self.train_stream, batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_stream, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader

    def to_dict(self):
        return {
            'name': self.name,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'num_classes': len(self.class_order_train),
            'class_order_train': self.class_order_train,
            'class_order_test': self.class_order_test,
            'train_transform': repr(getattr(self.train_dataset, 'transform', None)),
            'test_transform': repr(getattr(self.test_dataset, 'transform', None))
        }
