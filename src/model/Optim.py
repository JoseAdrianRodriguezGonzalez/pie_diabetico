import torch.optim as optim
def get_optimizer(model,lr=1e-4):
    return optim.Adam(params=model.classifier.parameters(),
                      lr=lr)
