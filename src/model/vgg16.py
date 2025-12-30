from torchvision import datasets, transforms
from torch.utils.data import Subset 
from sklearn.model_selection import train_test_split
import numpy as np


preprocess=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )]
)
data=datasets.ImageFolder(
    root='../data/datasets_diabetes/DFUNET/PartA_DFU_Dataset/',
    transform=preprocess
)
labels=[label for _,label in data.samples]
print(labels)

indices = np.arange(len(data))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
train_dataset=Subset(data,train_idx)
test_dataset=Subset(data,test_idx)
