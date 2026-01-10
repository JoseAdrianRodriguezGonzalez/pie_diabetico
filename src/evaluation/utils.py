from torch.utils.data import DataLoader
def make_train_stage():
    return {"epochs_index":0,
            "train_loss":[],
            "train_acc":[],
            "test_loss":[],
            "test_acc":[]}

