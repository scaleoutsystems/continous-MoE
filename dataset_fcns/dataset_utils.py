import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt


def ensure_cifar10(root: str, download: bool = True):
    """Download CIFAR-10 dataset if not already present."""
    if not os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
        torchvision.datasets.CIFAR10(root=root, train=True, download=download)
        torchvision.datasets.CIFAR10(root=root, train=False, download=download)
    return


def ensure_imagenet(root: str):
    """Make sure ImageNet data exists under ``root``.

    We do **not** attempt to download ImageNet automatically; the user must
    place a copy of the dataset (e.g. ILSVRC2012) into the datasets folder.
    If the expected train/val subdirectories are missing we raise an error
    with instructions.
    """
    # common layout: root/imagenet/train and root/imagenet/val or
    # root/ILSVRC2012_img_train
    candidates = [
        os.path.join(root, 'imagenet'),
        os.path.join(root, 'ILSVRC2012_img_train'),
    ]
    exists = any(os.path.isdir(c) for c in candidates)
    if not exists:
        raise RuntimeError(
            "ImageNet data not found under dataset_root. "
            "Please download ImageNet yourself and place the directory in '"
            f"{root}' (e.g. create '{root}/imagenet')."
        )


def ensure_imagenette(root: str):
    """Verify that Imagenette data is available."""
    # dataset repo uses folder named imagenette2
    imnet_root = os.path.join(root, 'imagenette2')
    if not os.path.isdir(imnet_root):
        raise RuntimeError(
            "Imagenette directory not found under dataset_root. "
            "Please download Imagenette and unpack it to '"
            f"{imnet_root}'."
        )


def ensure_core50(root: str):
    """Check for Core50 dataset.

    The expected layout is ``root/core50`` but the loader does not inspect
    the internals. The user must supply the dataset manually.
    """
    core_root = os.path.join(root, 'core50')
    if not os.path.isdir(core_root):
        raise RuntimeError(
            "Core50 directory not found under dataset_root. "
            "Please download Core50 and place it in '" f"{core_root}'."
        )


def make_mini_cifar(root: str, num_samples: int, seed: int | None = 0,
                    train: bool = True, transform=None):
    """Return a small subset of CIFAR-10 containing *num_samples* images.

    Samples are chosen uniformly at random from the train or test split.
    """
    ensure_cifar10(root)
    if transform is None:
        transform = transforms.Compose([
                    transforms.ToTensor(),
        ])
    full = torchvision.datasets.CIFAR10(root=root, train=train,
                                        download=False, transform=transform)
    rng = np.random.RandomState(seed) if (seed is not None and seed != 0) else np.random
    indices = rng.choice(len(full), size=num_samples, replace=False).tolist()
    # convert to plain list (type checker sometimes complains)
    return Subset(full, list(indices))  # type: ignore


def dirichlet_split(
    targets,
    num_partitions: int,
    alpha: float,
    min_size: int = 10,
    balanced: bool = False,
    seed: int | None = 0,
    max_attempts: int = 10,
) -> List[List[int]]:
    """Split indices according to a Dirichlet distribution.

    This is the same implementation copied/adapted from the project notebook
    and returns a list of index lists, one per partition.
    """
    if seed is not None and seed != 0:
        np.random.seed(seed)
    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    N = len(targets)

    for attempt in range(max_attempts):
        class_indices = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(targets):
            class_indices[int(label)].append(idx)

        partitions = [[] for _ in range(num_partitions)]

        if balanced:
            # allocate roughly equal sized chunks
            partition_size = N // num_partitions
            leftover = N - partition_size * num_partitions

            for c in range(num_classes):
                indices = np.array(class_indices[c])
                np.random.shuffle(indices)
                proportions = np.random.dirichlet(alpha * np.ones(num_partitions))
                counts = (proportions * len(indices)).astype(int)
                while counts.sum() < len(indices):
                    counts[np.argmax(counts)] += 1
                start = 0
                for i in range(num_partitions):
                    partitions[i].extend(indices[start:start + counts[i]])
                    start += counts[i]

            # trim and redistribute leftovers
            for i in range(num_partitions):
                if len(partitions[i]) > partition_size:
                    partitions[i] = partitions[i][:partition_size]
            leftovers = []
            for i in range(num_partitions):
                leftovers.extend(partitions[i][partition_size:])
            np.random.shuffle(leftovers)
            for i in range(leftover):
                partitions[i].append(leftovers.pop())
        else:
            for c in range(num_classes):
                indices = np.array(class_indices[c])
                np.random.shuffle(indices)
                proportions = np.random.dirichlet(alpha * np.ones(num_partitions))
                counts = (proportions * len(indices)).astype(int)
                while counts.sum() < len(indices):
                    counts[np.argmax(counts)] += 1
                start = 0
                for i in range(num_partitions):
                    partitions[i].extend(indices[start:start + counts[i]])
                    start += counts[i]

        sizes = [len(p) for p in partitions]
        if min(sizes) >= min_size:
            return partitions

    raise RuntimeError(f"Failed to satisfy min_size={min_size} after "
                       f"{max_attempts} attempts")


def static_split(indices: List[int], num_partitions: int, seed: int | None = 0) -> List[List[int]]:
    """Split a list of indices into *num_partitions* contiguous chunks.

    Alternates according to a shuffled permutation so class-balance is preserved
    when the input has randomized order.
    """
    rng = np.random.RandomState(seed) if seed is not None and seed != 0 else np.random
    arr = np.array(indices)
    rng.shuffle(arr)
    length = len(arr)
    part_size = length // num_partitions
    out = []
    for i in range(num_partitions):
        start = i * part_size
        end = (i + 1) * part_size if i < num_partitions - 1 else length
        out.append(arr[start:end].tolist())
    return out


def create_train_test_loaders(
    dataset,
    partition_indices: List[List[int]],
    train_frac: float = 0.8,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int | None = 0,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Given a base dataset and partitions, return lists of train/test loaders.

    The dataset is split into train/test for each partition and DataLoader objects
    are constructed. Shuffling/randomness is seeded for reproducibility.
    """
    # only seed if a nonzero value was provided; 0 means "random"
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_loaders = []
    test_loaders = []
    for i, part in enumerate(partition_indices):
        part = np.array(part)
        # # sanity check: make sure all indices are within dataset range
        # if len(part) > 0 and max(part) >= len(dataset):
        #     print(f"[DEBUG] partition {i} contains max index {max(part)} >= dataset length {len(dataset)}")
        np.random.shuffle(part)
        split_idx = int(len(part) * train_frac)
        train_idx, test_idx = part[:split_idx], part[split_idx:]

        # convert to plain Python lists for Subset compatibility
        train_idx = list(train_idx)
        test_idx = list(test_idx)

        # # inspect train/test splits as well
        # if len(train_idx) > 0 and max(train_idx) >= len(dataset):
        #     print(f"[DEBUG] train split of partition {i} invalid index {max(train_idx)}")
        # if len(test_idx) > 0 and max(test_idx) >= len(dataset):
        #     print(f"[DEBUG] test split of partition {i} invalid index {max(test_idx)}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loaders.append(
            DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
        )
        test_loaders.append(
            DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
        )
    return train_loaders, test_loaders


def create_dataloaders(config: Dict) -> Dict:
    """Create datasets and dataloaders according to a configuration dictionary.

    Returns a dict containing the following keys:
      - "dataset": the base dataset object
      - "train_loaders": list of train DataLoaders
      - "test_loaders": list of test DataLoaders
      - "pretrain_loader": optional DataLoader for pretraining (or None)
      - "partition_type": string used ("dirichlet", "static" or "random")

    The config dictionary is expected to have entries described in the JSON
    example (see configs/example_config.json).
    """
    root = config.get("dataset_root", "./datasets")
    name = config.get("dataset", "cifar10")
    mini = config.get("mini_dataset", None)

    # seed handling: allow finer control via config['seeds']
    seeds = config.get("seeds", {})
    # helper that interprets 0 or missing as "no seeding"
    def _norm(s):
        if s is None:
            return None
        if isinstance(s, str) and s.lower() == "random":
            return None
        try:
            iv = int(s)
            if iv == 0:
                return None
            return iv
        except Exception:
            return None
    global_seed = _norm(seeds.get("global", config.get("seed", None)))
    dataset_seed = _norm(seeds.get("dataset", global_seed))
    partition_seed = _norm(seeds.get("partition", dataset_seed))
    loader_seed = _norm(seeds.get("loader", dataset_seed))
    pretrain_seed = _norm(seeds.get("pretrain", dataset_seed))
    # seed the various RNGs before dataset construction if requested
    if dataset_seed is not None:
        np.random.seed(dataset_seed)
        random.seed(dataset_seed)
        torch.manual_seed(dataset_seed)
    
    resize = config.get("resize", 0)
    if resize > 0:
        trf = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    else: 
        trf = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load base dataset
    lname = name.lower()
    if lname == "cifar10":
        if mini is not None:
            dataset = make_mini_cifar(root, mini, dataset_seed, train=True, transform=trf)
        else:
            ensure_cifar10(root)
            dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                   download=False, transform=trf)
    elif lname == "imagenet":
        ensure_imagenet(root)
        # ImageFolder will read whatever images are in the train directory
        img_root = os.path.join(root, "imagenet")
        if os.path.isdir(os.path.join(img_root, "train")):
            img_root = os.path.join(img_root, "train")
        dataset = torchvision.datasets.ImageFolder(img_root, transform=trf)
    elif lname == "imagenette":
        ensure_imagenette(root)
        img_root = os.path.join(root, "imagenette2")
        if os.path.isdir(os.path.join(img_root, "train")):
            img_root = os.path.join(img_root, "train")
        dataset = torchvision.datasets.ImageFolder(img_root, transform=trf)
    elif lname == "core50":
        ensure_core50(root)
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, "core50"), transform=trf)
    else:
        raise ValueError(f"Unsupported dataset {name}")

    # partitioning
    num_parts = config.get("num_partitions", 1)
    partition_params = config.get("partition", {})
    p_type = partition_params.get("type", "random")
    if p_type == "dirichlet":
        alpha = partition_params.get("alpha", 0.5)
        min_size = partition_params.get("min_size", 1)
        balanced = partition_params.get("balanced", False)
        # obtain the list of labels corresponding to the current dataset
        if isinstance(dataset, Subset):
            # Subset may wrap a larger dataset; we need labels only for the
            # indices retained in the subset to avoid out-of-bounds partitioning.
            targets = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
        elif hasattr(dataset, 'targets'):
            targets = dataset.targets  # type: ignore
        else:
            # general fallback (should rarely be reached)
            targets = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
        indices_lists = dirichlet_split(targets,
                                        num_partitions=num_parts,
                                        alpha=alpha,
                                        min_size=min_size,
                                        balanced=balanced,
                                        seed=partition_seed)
    elif p_type == "static":
        indices = list(range(len(dataset)))
        indices_lists = static_split(indices, num_parts, seed=partition_seed)
    else:  # random or single loader
        if num_parts == 1:
            indices_lists = [list(range(len(dataset)))]
        else:
            indices_lists = static_split(list(range(len(dataset))), num_parts, seed=partition_seed)
        p_type = "random" if num_parts == 1 else "static"

    train_frac = config.get("train_frac", 0.8)
    batch_size = config.get("batch_size", 128)
    shuffle = config.get("shuffle", True)

    train_loaders, test_loaders = create_train_test_loaders(
        dataset,
        indices_lists,
        train_frac=train_frac,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=loader_seed,
    )

    pretrain_loader = None
    pretrain_cfg = config.get("pretrain", {})
    if pretrain_cfg.get("enabled", False):
        num_pre = pretrain_cfg.get("num_samples", 0)
        if num_pre > 0:
            # draw samples from the *train* portion of the full dataset
            all_indices = list(range(len(dataset)))
            random.seed(pretrain_seed)
            pre_indices = random.sample(all_indices, min(num_pre, len(all_indices)))
            pre_subset = Subset(dataset, pre_indices)
            pretrain_loader = DataLoader(pre_subset, batch_size=batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)
            # optionally remove pretrain samples from future training set
            if not pretrain_cfg.get("with_replacement", False):
                # remove those samples from each partition loader
                for i, part in enumerate(indices_lists):
                    indices_lists[i] = [idx for idx in part if idx not in pre_indices]
                train_loaders, test_loaders = create_train_test_loaders(
                    dataset,
                    indices_lists,
                    train_frac=train_frac,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=loader_seed,
                )

    return {
        "dataset": dataset,
        "train_loaders": train_loaders,
        "test_loaders": test_loaders,
        "pretrain_loader": pretrain_loader,
        "partition_type": p_type,
        "train_frac": train_frac,
        "batch_size": batch_size,
    }


def plot_partition_distributions(train_loaders, test_loaders, num_classes=10, class_names=None):
    """Visualize class counts per partition (train/test/overall).

    This helper avoids iterating DataLoader objects directly since worker
    failures can occur for small subsets (e.g. mini dataset) when a batch
    straddles the boundary.  Instead we inspect each loader's underlying
    ``Subset`` indices and fetch labels from the base dataset.
    """

    num_partitions = len(train_loaders)
    fig, axes = plt.subplots(3, num_partitions, figsize=(4*num_partitions, 12))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # helper to gather labels from a loader without iterating it
    def _get_labels(loader):
        ds = loader.dataset
        # Subset wraps the real dataset
        if hasattr(ds, "indices"):
            inds = ds.indices
            base = ds.dataset
        else:
            # fallback: iterate once (should not happen)
            labels = []
            for _, lab in loader:
                labels.extend(lab.numpy())
            return labels
        labels = []
        for idx in inds:
            # idx may be numpy scalar
            idx = int(idx)
            labels.append(base[idx][1])
        return labels

    for i in range(num_partitions):
        train_counts = [0] * num_classes
        test_counts = [0] * num_classes

        train_labels = _get_labels(train_loaders[i])
        for lbl in train_labels:
            train_counts[lbl] += 1
        test_labels = _get_labels(test_loaders[i])
        for lbl in test_labels:
            test_counts[lbl] += 1

        overall_counts = np.array(train_counts) + np.array(test_counts)

        axes[0, i].bar(class_names, train_counts, color='skyblue')
        axes[0, i].set_title(f"Partition {i} Train")
        axes[0, i].set_ylim(0, max(overall_counts)*1.1)

        axes[1, i].bar(class_names, test_counts, color='lightgreen')
        axes[1, i].set_title(f"Partition {i} Test")
        axes[1, i].set_ylim(0, max(overall_counts)*1.1)

        axes[2, i].bar(class_names, overall_counts, color='salmon')
        axes[2, i].set_title(f"Partition {i} Overall")
        axes[2, i].set_ylim(0, max(overall_counts)*1.1)

        overall_counts = np.array(train_counts) + np.array(test_counts)

        axes[0, i].bar(class_names, train_counts, color='skyblue')
        axes[0, i].set_title(f"Partition {i} Train")
        axes[0, i].set_ylim(0, max(overall_counts)*1.1)

        axes[1, i].bar(class_names, test_counts, color='lightgreen')
        axes[1, i].set_title(f"Partition {i} Test")
        axes[1, i].set_ylim(0, max(overall_counts)*1.1)

        axes[2, i].bar(class_names, overall_counts, color='salmon')
        axes[2, i].set_title(f"Partition {i} Overall")
        axes[2, i].set_ylim(0, max(overall_counts)*1.1)

        for ax in axes[:, i]:
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

