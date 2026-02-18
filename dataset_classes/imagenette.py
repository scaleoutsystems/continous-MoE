"""Imagenette continual-dataset class (wrapper around torchvision.datasets.Imagenette)."""
from torchvision import transforms
import torchvision

from .continual_base import ContinualDataset


class ImagenetteContinual(ContinualDataset):
    def __init__(self, root, desired_size=(224, 224), download=False, batch_size=1, num_workers=0,
                 pretrain_samples_per_class: int = 0, pretrain_reuse: bool = False,
                 shuffle_within_class: bool = False, seed: int = None):
        resize_transform = transforms.Compose([
            transforms.Resize(desired_size),
            transforms.ToTensor()
        ])

        dataset_train = torchvision.datasets.Imagenette(root=root, split='train', download=download, transform=resize_transform)
        dataset_test = torchvision.datasets.Imagenette(root=root, split='val', download=download, transform=resize_transform)

        super().__init__('imagenette', dataset_train, dataset_test, batch_size=batch_size, num_workers=num_workers,
                         pretrain_samples_per_class=pretrain_samples_per_class,
                         pretrain_reuse=pretrain_reuse,
                         shuffle_within_class=shuffle_within_class,
                         seed=seed)


def get_imagenette(root, **kwargs):
    return ImagenetteContinual(root, **kwargs)
