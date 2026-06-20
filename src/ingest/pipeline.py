from torchvision import transforms
from ingest.discover import images
from ingest.split import stratified
from ingest.dataset import DFU
from ingest.dataloader import make_loader
from preprocess.filters import dfu_transforms,dfu_transforms_train
def pipeline_ingest(src,args):
    diccionario=images(src)
    diccionario_formato={"images":[img for label,images in diccionario.items() for img in images],
                         "labels":[label for label,images in diccionario.items() for _ in images]} 
    datos_divididos=stratified(**diccionario_formato,seed=args.seed,val_size=0.1)
    label_to_index = {
#        "Abnormal": 1,
#        "Normal": 0
        "Aug-Negative": 1,
        "Aug-Positive": 0
    

    }
    train_df=DFU("train",datos_divididos,dfu_transforms(),label_to_index)
    val_df=DFU("val",datos_divididos,dfu_transforms(),label_to_index)
    test_df=DFU("test",datos_divididos,dfu_transforms(),label_to_index)
    train_loader=make_loader(train_df,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers,
                             drop_last=args.drop_last,
                             pin_memory=args.pin_memory)
    val_loader=make_loader(val_df,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           drop_last=False,
                           pin_memory=args.pin_memory
                           )

    index_to_label = {i: l for l, i in label_to_index.items()}
    test_loader=make_loader(test_df,batch_size=args.batch_size,
                            shuffle=False,num_workers=args.num_workers,drop_last=False,pin_memory=args.pin_memory)
    return train_loader,val_loader,test_loader,index_to_label
