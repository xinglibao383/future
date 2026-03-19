from torch.utils.data import DataLoader, random_split
from utils.xrfv22 import *
from utils.xrfv22_max_predict_len import *


def get_dataloaders_v3(root_path, use_len, compute_len, predict_len, stride_len, batch_size, train_ratio, exclude_device_idx=None, cross=None, cross_idx=None, mode=None, random_seed=None):
    if random_seed != None: torch.manual_seed(random_seed)
    dataset= XRFV22(root_path, use_len, compute_len, predict_len, stride_len, exclude_device_idx, cross, cross_idx, mode)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


def get_dataloaders_v3_cross_experiment(root_path, use_len, compute_len, predict_len, stride_len, batch_size, cross, cross_idx):
    train_dataset= XRFV22(root_path, use_len, compute_len, predict_len, stride_len, cross=cross, cross_idx=cross_idx, mode="train")
    val_dataset= XRFV22(root_path, use_len, compute_len, predict_len, stride_len, cross=cross, cross_idx=cross_idx, mode="val")
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


def get_dataloaders_v3_max_predict_len(root_path, use_len, compute_len, predict_len, stride_len, batch_size, train_ratio, random_seed=3407):
    torch.manual_seed(random_seed)
    dataset= XRFV22MaxPredictLen(root_path, use_len, compute_len, predict_len, stride_len)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


if __name__ == "__main__":
    pass