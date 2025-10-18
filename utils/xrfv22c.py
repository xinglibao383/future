import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.data_process4_c import *


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
        label = int(os.path.splitext(os.path.basename(filepath))[0].split("_")[-1])
        return imu, label
