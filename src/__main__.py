from ingest.pipeline import pipeline_ingest
from model.pipeline import pipeline_models
from evaluation.pipeline import complete_routine
from argparse import Namespace
import torch 
from utils.plots import curves,plot_confussion_matrix
from utils.make_folders import make_dir_test
import os 
from tqdm import tqdm 
def main(args):
    for i in range(args.test_number):
        path=os.path.join(args.src,str(i))
        os.makedirs(path,exist_ok=True)
        make_dir_test(path,args.models,args.submodels)
    src= "../data/raw/datasets_diabetes/DFUNET/PartA_DFU_Dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in tqdm(range(args.test_number),desc="Tests"):
        path_file=os.path.join(args.src,str(i)) 
        seed=args.seed+i
        args.seed=seed 
        df_train,df_val,df_test,index_to_label=pipeline_ingest(src,args)
        classes = [index_to_label[i] for i in range(len(index_to_label))]
        for m in tqdm(args.models,desc=f"models (test {i})",leave=False):
            if len(args.submodels[m])==0:
                path_file_model=os.path.join(path_file,m)
                model,loss,optimizer=pipeline_models(m,"")
                model_name=f"{m}"
            
                model=model.to(device)
                train_state=complete_routine(args,df_train,df_val,df_test,model,loss,optimizer,device,path_file_model)
                for keys, values in train_state.items():
                    print(f"{keys}: {values}")
                curves((train_state["train_acc"],train_state["val_acc"]),("train_acc","val_acc"),model_name=model_name+"_accuracy",path=path_file_model)
                curves((train_state["train_loss"],train_state["val_loss"]),("train_loss","val_loss"),model_name=model_name+"_loss",path=path_file_model)
                curves((train_state["train_f1_score"],train_state["val_f1_score"]),("train_f1_score","val_f1_score"),model_name=model_name+"_f1_score",path=path_file_model)
                plot_confussion_matrix(train_state["cm"],classes,model_name,path_file_model)
            else:
                for s in tqdm(args.submodels[m],desc="submodels"):
                    path_file_model=os.path.join(path_file,m,s)
                    model,loss,optimizer=pipeline_models(m,s)
                    model_name=f"{m}_{s}"

                    model=model.to(device)
                    train_state=complete_routine(args,df_train,df_val,df_test,model,loss,optimizer,device,path_file_model)
                    for keys, values in train_state.items():
                        print(f"{keys}: {values}")
                    curves((train_state["train_acc"],train_state["val_acc"]),("train_acc","val_acc"),model_name=model_name+"_accuracy",path=path_file_model)
                    curves((train_state["train_loss"],train_state["val_loss"]),("train_loss","val_loss"),model_name=model_name+"_loss",path=path_file_model)
                    curves((train_state["train_f1_score"],train_state["val_f1_score"]),("train_f1_score","val_f1_score"),model_name=model_name+"_f1_score",path=path_file_model)
                    plot_confussion_matrix(train_state["cm"],classes,model_name,path_file_model)


if __name__=="__main__":
    args=Namespace(num_epochs=16,   
                   batch_size=4,
                    shuffle=True,
                    num_workers=6,
                    drop_last=True,
                    pin_memory=True,
                   test_number=10,
                   models=["efficientnet","vgg","resnet","googlenet","alexnet"],
                   submodels={"efficientnet":["B0","B1","B2","B3","B4","B5","B6","B7"],
                              "vgg":["11","13","16","19"],
                              "resnet":["34","50","18","101","152"],
                              "googlenet":[],
                              "alexnet":[]
                              },
                   src="../out/tests",
                   seed=42
                )
    main(args)
