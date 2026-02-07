from torch.utils.data import DataLoader
def make_train_stage():
    return {"epochs_index":0,
            "train_acc":[],
            "test_acc":[],
            "val_acc":[],
            "train_loss":[],
            "val_loss":[],
            "test_loss":[],
            }

