import torch
from sklearn.metrics import confusion_matrix
def compute_accuracy(y_pred,y_target):
    _,y_pred_indices=y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices,y_target).sum().item()
    return n_correct/y_target.size(0)
def compute_f1(y_pred,y_target):
    _,y_pred_indices = y_pred.max(dim=1)

    tp = ((y_pred_indices==1) & (y_target==1)).sum().item()
    fp = ((y_pred_indices==1) & (y_target==0)).sum().item()
    fn = ((y_pred_indices==0) & (y_target==1)).sum().item()

    precision=tp /(tp+fp +1e-8)
    recall = tp/(tp+fn+1e-8)
    f1=2 *precision*recall/(precision+recall+1e-8)
    return f1
def matriz(y_pred,y_target):
    return confusion_matrix(y_target,y_pred)
