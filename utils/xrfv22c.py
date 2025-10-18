import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
# from utils.data_process4_c import *
from data_process4_c import *


class XRFV22C(Dataset):
    def __init__(self, root_path, window_size, stride):
        super().__init__()
        self.imu_root_path = os.path.join(root_path, "imu", f"{window_size}_{stride}")
        if not os.path.exists(self.imu_root_path):
            process_raw_imu_data(
                data_path="/mnt/mydata/yh/liming/data/xrfv2/imu", 
                save_path="/mnt/mydata/yh/liming/workspace/future/mydatac/imu", 
                window_size=window_size, 
                stride=stride
            )
        self.imu_filepaths = [os.path.join(self.imu_root_path, f) for f in os.listdir(self.imu_root_path)]

    def __len__(self):
        return len(self.imu_filepaths)

    def __getitem__(self, idx):
        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)
        label = int(os.path.splitext(os.path.basename(self.imu_filepaths[idx]))[0].split("_")[-1])
        return imu, label


def get_dataloaders(root_path, window_size, stride, batch_size, train_ratio=0.8):
    dataset= XRFV22C(root_path, window_size, stride)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/utils/xrfv22c.py
if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders("/mnt/mydata/yh/liming/workspace/future/mydatac", 45, 15, 32, 0.8)
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)