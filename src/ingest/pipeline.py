from torchvision import transforms
from ingest.discover import images
from ingest.split import stratified
from ingest.dataset import DFU
from ingest.dataloader import make_loader
from preprocess.filters import dfu_transforms

DATASET_PARTA= "../data/raw/datasets_diabetes/DFUNET/PartA_DFU_Dataset/"
diccionario=images(DATASET_PARTA)
diccionario_formato={"images":[img for label,images in diccionario.items() for img in images],
                     "labels":[label for label,images in diccionario.items() for _ in images]}
datos_divididos=stratified(**diccionario_formato)
diccionario_formato={"train":{"images":datos_divididos[0],
                              "labels":datos_divididos[2]},
                     "test":{"images":datos_divididos[1],
                             "labels":datos_divididos[3]}}
df=DFU("train",diccionario_formato,dfu_transforms())
df_load=make_loader(df)
