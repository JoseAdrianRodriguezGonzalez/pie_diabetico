import torch
from sklearn.metrics import confusion_matrix
def compute_accuracy(y_pred,y_target):
    _,y_pred_indices=y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices,y_target).sum().item()
    return n_correct/y_target.size(0)
def compute_f1(y_pred,y_target):
    
    f1_scores=[]
    for cls in [0,1]:

        tp = ((y_pred==cls) & (y_target==cls)).sum().item()
        fp = ((y_pred==cls) & (y_target!=cls)).sum().item()
        fn = ((y_pred!=cls) & (y_target==cls)).sum().item()
        

        precision=tp /(tp+fp +1e-8)
        recall = tp/(tp+fn+1e-8)
        f1=2 *precision*recall/(precision+recall+1e-8)
        f1_scores.append(f1)
    return sum(f1_scores)/len(f1_scores)
def matriz(y_pred,y_target):
    return confusion_matrix(y_target,y_pred)
