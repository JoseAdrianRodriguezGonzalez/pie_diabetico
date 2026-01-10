from ingest.pipeline import pipeline_ingest
from model.pipeline import pipeline_models
from evaluation.pipeline import complete_routine
from argparse import Namespace
import torch 
def main(args):
    src= "../data/raw/datasets_diabetes/DFUNET/PartA_DFU_Dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train,df_test=pipeline_ingest(src,args)
    model,loss,optimizer=pipeline_models()
    model=model.to(device)
    complete_routine(args,df_train,df_test,model,loss,optimizer,device)
if __name__=="__main__":
    args=Namespace(num_epochs=50,   
                   batch_size=10,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True,
                    pin_memory=True)
    main(args)
