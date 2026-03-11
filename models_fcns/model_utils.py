import torch.nn as nn
import torchvision
from typing import Dict

# import the custom MoE ViT factory if available
try:
    from models_classes.moe_vit import create_moe_vit
except ImportError:
    create_moe_vit = None


def create_model(config: Dict):
    """Instantiate a model described by the configuration dictionary.

    The config should include an entry "model" which itself is a dict
    containing a "name" (e.g. "convnext_tiny", "resnet18", "vit_moe") 
    and additional parameters specific to that family. 
    
    If resnet has "-cifar" suffix, it will use a version optimized 
    for CIFAR-10.

    Returns the created torch.nn.Module.
    """
    mcfg = config.get("model", {})
    name = mcfg.get("name", "convnext_tiny")
    name = name.lower()

    if name.startswith("convnext"):
        # convnext architectures exposed by torchvision
        fn = getattr(torchvision.models, name, None)
        if fn is None:
            raise ValueError(f"Unknown convnext model {name}")
        model = fn(weights=mcfg.get("pretrained", False))
        num_classes = mcfg.get("num_classes", 10)
        if hasattr(model, 'classifier'):
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        elif hasattr(model, 'head'):
            model.head = nn.Linear(model.head.in_features, num_classes)
        # adapt first conv for small images if image size is known
        img_size = mcfg.get('img_size', None)
        if img_size is not None and img_size < 64:
            # replace the initial downsampling conv to use a 3x3 stride1 kernel
            if hasattr(model, 'features') and len(model.features) > 0:
                first = model.features[0]
                # Conv2dNormActivation stores the conv as first[0]
                if isinstance(first, nn.Sequential) and len(first) > 0 and isinstance(first[0], nn.Conv2d):
                    in_ch = first[0].in_channels
                    out_ch = first[0].out_channels
                    model.features[0][0] = nn.Conv2d(in_ch, out_ch,
                                                      kernel_size=3, stride=1, padding=1,
                                                      bias=False)
            # no further structural changes required; downstream stages will
            # simply operate on a larger spatial map.
        return model

    elif name.startswith("resnet"):
        # we adjust for small inputs automatically rather than relying on
        # a "-cifar" suffix.  older configs remain compatible.
        cifarVersion = False
        if name.endswith("-cifar"):
            name = name[:-6]
            cifarVersion = True
        fn = getattr(torchvision.models, name, None)
        if fn is None:
            raise ValueError(f"Unknown resnet model {name}")
        model = fn(weights=mcfg.get("pretrained", False))
        num_classes = mcfg.get("num_classes", 10)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        img_size = mcfg.get('img_size', None)
        if cifarVersion or (img_size is not None and img_size < 64):
            # Replace 7x7 conv with 3x3 stride 1
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove maxpool
            model.maxpool = nn.Identity()
            # Fix classifier again just in case
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "vit_moe" or name == "moe_vit":
        if create_moe_vit is None:
            raise ImportError("MoE ViT factory not found; ensure models_classes is on PYTHONPATH")
        p = mcfg.copy()
        # remove the name field, not used by factory
        p.pop("name", None)
        return create_moe_vit(**p)["model"]

    else:
        raise ValueError(f"Unsupported model name {name}")
