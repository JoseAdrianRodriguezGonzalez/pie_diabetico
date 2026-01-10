import torch 
import torch.nn as nn 
from torchvision import models
def get_model():
    vgg=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.classifier[6] = nn.Linear(4096, 2)
    for param in vgg.features.parameters():
        param.requires_grad = False
    return vgg

