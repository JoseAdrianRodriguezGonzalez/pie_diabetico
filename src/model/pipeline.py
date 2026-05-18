from sys import version
from model.Optim import get_optimizer
from model.lossess import get_function
from model.vgg16 import get_swin, get_vgg,get_efficientnet,get_resnet,get_alexnet,get_googlenet, get_vit
def pipeline_models(models,submodels):
    f=get_function()
    model=get_alexnet()
    if models=="efficientnet":
        model=get_efficientnet(version=int(submodels[-1]))
    elif models=="vgg":
        model=get_vgg(version=int(submodels))
    elif models=="resnet":
        model=get_resnet(version=int(submodels))
    elif models=="alexnet":
        model=get_alexnet()
    elif models=="googlenet":
        model=get_googlenet()
    elif models=="vit":
        model=get_vit(version=submodels)
    elif models=="swin":
        model=get_swin(version=submodels)
    optimizer_=get_optimizer(model)
    return model,f,optimizer_
