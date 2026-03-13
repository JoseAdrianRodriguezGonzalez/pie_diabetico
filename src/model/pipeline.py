from sys import version
from model.Optim import get_optimizer
from model.lossess import get_function
from model.vgg16 import get_vgg,get_efficientnet,get_resnet,get_alexnet,get_googlenet
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
    optimizer_=get_optimizer(model)
    return model,f,optimizer_
