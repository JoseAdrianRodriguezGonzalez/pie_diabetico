from model.Optim import get_optimizer
from model.lossess import get_function
from model.vgg16 import get_vgg,get_efficientnet
def pipeline_models():
    model=get_efficientnet(version=0)
    f=get_function()
    optimizer_=get_optimizer(model=model)
    return model,f,optimizer_
