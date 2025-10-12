import os
import torch
import numpy as np
from torch.utils.data import Dataset


class XRFV2(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        filenames = os.listdir(os.path.join(self.root_path, "imu"))
        filenames = [filename for filename in filenames if filename.startswith("15_1")]
        self.imu_filepaths = []
        self.pose_filepaths = []
        for filename in filenames:
            self.imu_filepaths.append(os.path.join(self.root_path, "imu", filename))
            self.pose_filepaths.append(os.path.join(self.root_path, "pose", filename))

    def __len__(self):
        return len(self.imu_filepaths)

    def __getitem__(self, idx):
        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32)
        return imu, pose