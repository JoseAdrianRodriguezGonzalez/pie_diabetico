from torch.utils.data import DataLoader
def make_loader(dataset,batch_size=16,shuffle=True,num_workers=4,drop_last=True,pin_memory=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
