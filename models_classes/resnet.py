"""
ResNet model factory compatible with the notebook's model-package interface.
Provides `create_resnet(...)` returning a dict with model, loss_fn, optimizer,
train_fn, test_fn, backward_fn, name and default_params.
"""
from collections import defaultdict
import torch
from torch import nn
import torchvision

# Reuse train/test/backward helpers from convnext module to keep behaviour consistent
from .convnext import train as _shared_train, test as _shared_test, backward_fn as _shared_backward


def create_resnet(arch='resnet18', num_classes=1000, device=None, lr=1e-3, pretrained=False):
    """Factory for ResNet variants.

    Args:
        arch: 'resnet18' or 'resnet50'
    """
    arch = arch.lower()
    if arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        # replace final fc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    if device is not None:
        model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return {
        'model': model,
        'loss_fn': loss,
        'optimizer': optimizer,
        'train_fn': _shared_train,
        'test_fn': _shared_test,
        'backward_fn': _shared_backward,
        'name': arch,
        'default_params': {'lr': lr, 'pretrained': pretrained, 'num_classes': num_classes}
    }
