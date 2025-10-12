import os
import torch
import numpy as np
from torch.utils.data import Dataset


class XRFV2(Dataset):
    def __init__(self, root_path, use_len, compute_len, predict_len, stride_len):
        super().__init__()
        self.root_path = root_path
        filenames = os.listdir(os.path.join(self.root_path, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}"))
        filenames = [filename for filename in filenames if len(filename.split('_')) > 1 and filename.split('_')[1] == "2"]
        self.imu_filepaths = []
        self.pose_filepaths = []
        for filename in filenames:
            self.imu_filepaths.append(os.path.join(self.root_path, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}", filename))
            self.pose_filepaths.append(os.path.join(self.root_path, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}", filename))

    def __len__(self):
        return len(self.imu_filepaths)

    def __getitem__(self, idx):
        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32)
        if imu.shape[1] == 0:
            imu = torch.zeros((30, 100))
        return imu, pose