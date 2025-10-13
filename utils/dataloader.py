from torch.utils.data import DataLoader, random_split
from utils.xrf55 import XRF55
from utils.xrfv2 import *
from utils.xrfv22 import *


def get_dataloaders(root_path, len_input, len_predict, batch_size, train_ratio):
    dataset= XRF55(root_path, len_input, len_predict)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


def get_dataloaders_v2(root_path, use_len, compute_len, predict_len, stride_len, batch_size, train_ratio):
    dataset= XRFV2(root_path, use_len, compute_len, predict_len, stride_len)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


def get_dataloaders_v3(root_path, use_len, compute_len, predict_len, stride_len, batch_size, train_ratio):
    dataset= XRFV22(root_path, use_len, compute_len, predict_len, stride_len)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders("/home/xinglibao/workspace/future/data", 60, 15, 32, 0.8)
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)