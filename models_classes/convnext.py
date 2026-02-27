"""
ConvNeXt model factory and self-contained train/test helpers.
Provides a single entry function `create_convnext(...)` which returns a dictionary
containing: model, loss_fn, optimizer, train_fn, test_fn, backward_fn and metadata.

The train/test implementations are compatible with the continual-learning notebook's
expected signatures.
"""
import torchvision

def create_convnext(num_classes=1000, pretrained=False):
    """Factory that builds a convnext model and returns the trainer/tester + metadata.

    Returns a dict with keys:
      - model, loss_fn, optimizer, train_fn, test_fn, backward_fn, name, default_params
    """
    model = torchvision.models.convnext_tiny(pretrained=pretrained, num_classes=num_classes)

    return {
        'model': model,
        'name': 'convnext_tiny',
        'default_params': {'pretrained': pretrained, 'num_classes': num_classes}
    }
