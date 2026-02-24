import torch.optim as optim
def get_classifier_params(model):
    if hasattr(model, "classifier"):
        return model.classifier.parameters()
    elif hasattr(model, "fc"):
        return model.fc.parameters()
    else:
        raise ValueError("No se encontró la capa final")

def get_optimizer(model,lr=1e-4):
    return optim.Adam(params=get_classifier_params(model),
                      lr=lr)
