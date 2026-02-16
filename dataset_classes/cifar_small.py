"""Tiny CIFAR-10 subset for fast debugging runs.

Usage:
  dataset = CIFAR10SmallContinual(root, samples_per_class_train=5, samples_per_class_test=5)

This creates torchvision CIFAR10 subsets with a small number of examples per class
and exposes the same interface as `ContinualDataset` so it plugs into the notebook
without further changes.
"""
from collections import defaultdict
import random

from torchvision import transforms
import torchvision
from torch.utils.data import Subset

from .continual_base import ContinualDataset


class CIFAR10SmallContinual(ContinualDataset):
    def __init__(self, root, desired_size=(224, 224), download=False,
                 batch_size=1, num_workers=0,
                 samples_per_class_train: int = 5,
                 samples_per_class_test: int = 5,
                 seed: int = 0):
        """Create very small per-class subsets of CIFAR-10 for fast tests.

        Args:
            samples_per_class_train: number of training samples to keep per class
            samples_per_class_test: number of test samples to keep per class
            seed: RNG seed for sampling (deterministic selection)
        """
        resize_transform = transforms.Compose([
            transforms.Resize(desired_size),
            transforms.ToTensor()
        ])

        # Load full CIFAR-10 datasets (transforms applied)
        full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=resize_transform)
        full_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=resize_transform)

        # Group indices by class for both splits
        train_by_class = defaultdict(list)
        for idx in range(len(full_train)):
            _, y = full_train[idx]
            train_by_class[int(y)].append(idx)

        test_by_class = defaultdict(list)
        for idx in range(len(full_test)):
            _, y = full_test[idx]
            test_by_class[int(y)].append(idx)

        rng = random.Random(seed)

        # Pick a small subset of indices per class
        selected_train = []
        for cls, inds in train_by_class.items():
            if len(inds) <= samples_per_class_train:
                chosen = inds.copy()
            else:
                chosen = rng.sample(inds, samples_per_class_train)
            selected_train.extend(chosen)

        selected_test = []
        for cls, inds in test_by_class.items():
            if len(inds) <= samples_per_class_test:
                chosen = inds.copy()
            else:
                chosen = rng.sample(inds, samples_per_class_test)
            selected_test.extend(chosen)

        # Build Subsets and hand off to the ContinualDataset base class
        train_subset = Subset(full_train, selected_train)
        test_subset = Subset(full_test, selected_test)

        super().__init__('cifar10_small', train_subset, test_subset, batch_size=batch_size, num_workers=num_workers)


def get_cifar10_small(root, **kwargs):
    return CIFAR10SmallContinual(root, **kwargs)
