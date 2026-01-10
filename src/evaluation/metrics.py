import torch
def compute_accuracy(y_pred,y_target):
    _,y_pred_indices=y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices,y_target).sum().item()
    return n_correct/y_target.size(0)
