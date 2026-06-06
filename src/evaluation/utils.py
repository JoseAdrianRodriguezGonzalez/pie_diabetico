from torch.utils.data import DataLoader
def make_train_stage():
    return {"epochs_index":0,
            "train_acc":[],
            "test_acc":[],
            "val_acc":[],
            "train_f1_score":[],
            "test_f1_score":[],
            "val_f1_score":[],
            "train_loss":[],
            "val_loss":[],
            "test_loss":[],
            "val_auc_roc":[],
            "val_auc_pr":[],
            "test_auc_roc":[],
            "test_auc_pr":[]
            }

