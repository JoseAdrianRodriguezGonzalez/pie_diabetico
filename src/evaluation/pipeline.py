from evaluation.utils import make_train_stage
from evaluation.train import train_routine,test_routine
from utils.write_info import write_info
def complete_routine(args,train_load,test_load,model,function,optimizer,device):
    train_state=make_train_stage()
    for epoch_index in range(args.num_epochs):
        train_state["epochs_index"]=epoch_index
        train_routine(train_state,train_load,model,function,optimizer,device)
        test_routine(train_state,test_load,model,function,device)
        print(f'Train loss:{train_state["train_loss"][epoch_index]}----- train accuracy: {train_state["train_acc"][epoch_index]}   |  test loss:  {train_state["test_loss"][epoch_index]}------- test accuracy: {train_state["test_acc"][epoch_index]}')
    #print(train_state)
    write_info("../out/accuracy.txt",train_state)
    

