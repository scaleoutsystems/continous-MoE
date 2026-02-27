"""
ResNet model factory compatible with the notebook's model-package interface.
Provides `create_resnet(...)` returning a dict with model, loss_fn, optimizer,
train_fn, test_fn, backward_fn, name and default_params.
"""
from torch import nn
import torchvision

# Reuse train/test/backward helpers from convnext module to keep behaviour consistent

def create_resnet(arch='resnet18', num_classes=1000, pretrained=False):
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

    return {
        'model': model,
        'name': arch,
        'default_params': {'pretrained': pretrained, 'num_classes': num_classes}
    }
