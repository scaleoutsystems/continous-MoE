"""CIFAR-10 continual-dataset class (wrapper around torchvision.datasets.CIFAR10).

Provides CIFAR10Continual which matches the interface of ImagenetteContinual.
"""
from torchvision import transforms
import torchvision

from .continual_base import ContinualDataset


class CIFAR10Continual(ContinualDataset):
    def __init__(self, root, desired_size=(224, 224), download=False, batch_size=1, num_workers=0):
        # Resize CIFAR images to the desired input size (ConvNeXt expects larger images)
        resize_transform = transforms.Compose([
            transforms.Resize(desired_size),
            transforms.ToTensor()
        ])

        dataset_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=resize_transform)
        dataset_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=resize_transform)

        # Name the dataset 'cifar10' so metadata is clear
        super().__init__('cifar10', dataset_train, dataset_test, batch_size=batch_size, num_workers=num_workers)


def get_cifar10(root, **kwargs):
    return CIFAR10Continual(root, **kwargs)
