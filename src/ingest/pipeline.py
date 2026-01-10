from torchvision import transforms
from ingest.discover import images
from ingest.split import stratified
from ingest.dataset import DFU
from ingest.dataloader import make_loader
from preprocess.filters import dfu_transforms
def pipeline_ingest(src,args):
    diccionario=images(src)
    diccionario_formato={"images":[img for label,images in diccionario.items() for img in images],
                         "labels":[label for label,images in diccionario.items() for _ in images]}
    datos_divididos=stratified(**diccionario_formato)
    diccionario_formato={"train":{"images":datos_divididos[0],
                                  "labels":datos_divididos[2]},
                         "test":{"images":datos_divididos[1],
                                 "labels":datos_divididos[3]}}
    train_df=DFU("train",diccionario_formato,dfu_transforms())
    test_df=DFU("test",diccionario_formato,dfu_transforms())
    train_loader=make_loader(train_df,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers,
                             drop_last=args.drop_last,
                             pin_memory=args.pin_memory)
    test_loader=make_loader(test_df,batch_size=args.batch_size,
                            shuffle=False,num_workers=args.num_workers,drop_last=False,pin_memory=args.pin_memory)
    return train_loader,test_loader
