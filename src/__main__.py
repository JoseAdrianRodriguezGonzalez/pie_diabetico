from ingest.pipeline import pipeline_ingest
from model.pipeline import pipeline_models
from evaluation.pipeline import complete_routine
from argparse import Namespace
import torch 
from utils.plots import curves,plot_confussion_matrix
def main(args):
    src= "../data/raw/datasets_diabetes/DFUNET/PartA_DFU_Dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train,df_val,df_test,index_to_label=pipeline_ingest(src,args)
    
    classes = [index_to_label[i] for i in range(len(index_to_label))]
    model,loss,optimizer=pipeline_models()
    model=model.to(device)
    train_state=complete_routine(args,df_train,df_val,df_test,model,loss,optimizer,device)
    for keys, values in train_state.items():
        print(f"{keys}: {values}")
    curves((train_state["train_acc"],train_state["val_acc"]),("train_acc","val_acc"))
    curves((train_state["train_loss"],train_state["val_loss"]),("train_loss","val_loss"))
    curves((train_state["train_f1_score"],train_state["val_f1_score"]),("train_f1_score","val_f1_score"))
    plot_confussion_matrix(train_state["cm"],classes,"vgg13","../out/vgg/vgg13/")
if __name__=="__main__":
    args=Namespace(num_epochs=16,   
                   batch_size=4,
                    shuffle=True,
                    num_workers=6,
                    drop_last=True,
                    pin_memory=True)
    main(args)
