import os
import torch
import numpy as np
from torch.utils.data import Dataset


class XRF55(Dataset):
    def __init__(self, root_path, len_input, len_predict):
        super().__init__()
        self.root_path = root_path
        self.len_input_imu = len_input * 50
        self.len_input_pose = len_input * 15
        self.len_predict_imu = len_predict * 50
        self.len_predict_pose = len_predict * 15
        self.window_size_imu = self.len_input_imu + self.len_predict_imu
        self.window_size_pose = self.len_input_pose + self.len_predict_pose
        self.stride_imu = self.len_predict_imu
        self.stride_pose = self.len_predict_pose
        self.data_path_imu = os.path.join(self.root_path, f"imu_{self.len_input_imu}_{self.len_predict_imu}")
        self.data_path_pose = os.path.join(self.root_path, f"pose_{self.len_input_pose}_{self.len_predict_pose}")
        self.filenames= self.get_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        imu = torch.tensor(np.load(os.path.join(self.data_path_imu, self.filenames[idx])), dtype=torch.float32)
        imu = imu.permute(1, 0, 2).reshape(-1, 5 * 6)
        pose = torch.tensor(np.load(os.path.join(self.data_path_pose, self.filenames[idx])), dtype=torch.float32)
        pose = pose[:, :, :2].reshape(-1, 25 * 2)
        return imu[:self.len_input_imu], imu[self.len_input_imu:], pose[:self.len_input_pose], pose[self.len_input_pose:]
    
    def get_filenames(self):
        filenames_imu = self.get_filenames_imu()
        filenames_pose = self.get_filenames_pose()
        return sorted(list(set(filenames_imu) & set(filenames_pose)))
    
    def get_filenames_imu(self):
        if not os.path.isdir(self.data_path_imu):
            os.makedirs(self.data_path_imu, exist_ok=True)
            for filename in os.listdir(os.path.join(self.root_path, "imu_raw")):
                data = np.load(os.path.join(self.root_path, "imu_raw", filename))
                num_windows = 0
                for start in range(0, data.shape[1] - self.window_size_imu + 1, self.stride_imu):
                    np.save(os.path.join(self.data_path_imu, f"{os.path.splitext(filename)[0]}_{num_windows}.npy"), data[:, start:start + self.window_size_imu, :])
                    num_windows += 1
        return os.listdir(self.data_path_imu)
    
    def get_filenames_pose(self):
        if not os.path.isdir(self.data_path_pose):
            os.makedirs(self.data_path_pose, exist_ok=True)
            for filename in os.listdir(os.path.join(self.root_path, "pose_raw")):
                data = np.load(os.path.join(self.root_path, "pose_raw", filename))
                num_windows = 0
                for start in range(0, data.shape[0] - self.window_size_pose + 1, self.stride_pose):
                    np.save(os.path.join(self.data_path_pose, f"{os.path.splitext(filename)[0]}_{num_windows}.npy"), data[start:start + self.window_size_pose])
                    num_windows += 1
        return os.listdir(self.data_path_pose)
    

