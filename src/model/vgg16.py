import torch 
import torch.nn as nn 
from torchvision import models
def get_vgg(
    *,
    version: int = 16,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_features: bool = True,
):
    weights = (
        models.VGG16_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    if version == 11:
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    elif version == 13:
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    elif version == 16:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif version == 19:
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("VGG version debe ser 11, 13, 16 o 19")
    model.classifier[6] = nn.Linear(4096, num_classes)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model

def get_efficientnet(
    *,
    version: int = 0,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_features: bool = True,
):
    weights_map = {
        0: models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        1: models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        2: models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        3: models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        4: models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        5: models.EfficientNet_B5_Weights.IMAGENET1K_V1,
        6: models.EfficientNet_B6_Weights.IMAGENET1K_V1,
        7: models.EfficientNet_B7_Weights.IMAGENET1K_V1,
    }

    constructors = {
        0: models.efficientnet_b0,
        1: models.efficientnet_b1,
        2: models.efficientnet_b2,
        3: models.efficientnet_b3,
        4: models.efficientnet_b4,
        5: models.efficientnet_b5,
        6: models.efficientnet_b6,
        7: models.efficientnet_b7,
    }

    if version not in constructors:
        raise ValueError("EfficientNet version debe estar entre 0 y 7")

    weights = weights_map[version] if pretrained else None
    model = constructors[version](weights=weights)

    # ---------- classifier ----------
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------- freeze ----------
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model
