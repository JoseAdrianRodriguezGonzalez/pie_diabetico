from torch.utils.data import Dataset
from PIL import Image
class DFU(Dataset):
    def __init__(self,mode,dfu_dict,transforms=None):
        self.transform=transforms
        self.DFU={}
        required={"train","test"}
        optional={"val"}
        missing=required-dfu_dict.keys()
        if missing:
            raise KeyError(f"Faltan splits obligatorios: {missing}")
        for split,data in dfu_dict.items():
            if split not in required | optional:
                raise KeyError(f"Split desconocido '{split}'")
            if not {"images","labels"}<=data.keys():
                raise KeyError(f"Split '{split}' mal formado")
            if len(data["images"])!=len(data["labels"]):
                raise KeyError("Desbalance de imagenes y labels en el split '{split}'")
        self.dfu=dfu_dict
        self.set_mode(mode)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image=Image.open(self.images[index]).convert("RGB")
        label=self.labels[index]
        if self.transform:
            image=self.transform(image)
        return image,label  
    def set_mode(self,mode):
        if mode not in self.dfu:
            raise KeyError("Modo '{mode}' no está")
        self.mode=mode
        self.images=self.dfu[mode]["images"]
        self.labels=self.dfu[mode]["labels"]
