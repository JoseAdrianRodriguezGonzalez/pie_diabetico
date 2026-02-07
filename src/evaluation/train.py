import torch
from ingest.dataloader import make_loader
from evaluation.metrics import compute_accuracy
def train_routine(train_state,data,model,function,optimizer,device):
    runing_loss=0.0
    running_accuracy=0.0
    model.train()
    for batch_index,(x,y) in enumerate(data):
        #Zero grad 
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_pred=model(x.float())
        loss=function(y_pred,y.long())
        loss_batch=loss.item()
        runing_loss+=(loss_batch- runing_loss)/(batch_index+1)
        #poas 4, los valores 
        loss.backward()
        #Paso 5 optimizador 
        optimizer.step()
        acc_batch= compute_accuracy(y_pred,y)
        running_accuracy+=(acc_batch- running_accuracy)/(batch_index+1)
    train_state["train_loss"].append(runing_loss)
    train_state["train_acc"].append(running_accuracy)
def eval_routine(train_state,data,model,function,device,prefix="val"):
    running_loss=0.0
    running_accuracy=0.0
    model.eval()
    with torch.no_grad():
        for batch_index,(x,y) in enumerate(data):
            x=x.to(device)
            y=y.to(device)
            y_pred=model(x.float())
            loss=function(y_pred,y.long())
            loss_batch=loss.item()
            running_loss+=(loss_batch- running_loss)/(batch_index+1) 
            acc_batch= compute_accuracy(y_pred,y)
            running_accuracy+=(acc_batch- running_accuracy)/(batch_index+1) 
        train_state[f"{prefix}_loss"].append(running_loss)
        train_state[f"{prefix}_acc"].append(running_accuracy) 
