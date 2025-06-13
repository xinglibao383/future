import os
import torch
import numpy as np
from torch.utils.data import Dataset


class XRF55(Dataset):
    def __init__(self, root_path, len_input, len_predict):
        self.root_path = root_path
        self.len_input = len_input
        self.len_predict = len_predict
        self.window_size = len_input + len_predict
        self.stride = len_predict
        self.data_path = os.path.join(self.root_path, f"pose_{self.len_input}_{self.len_predict}")
        self.filenames = self.get_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(os.path.join(self.data_path, self.filenames[idx])), dtype=torch.float32)
        return data[:self.len_input], data[self.len_input:]
    
    def get_filenames(self):
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
            for filename in os.listdir(os.path.join(self.root_path, "pose_raw")):
                data = np.load(os.path.join(self.root_path, "pose_raw", filename))
                num_windows = 0
                for start in range(0, data.shape[0] - self.window_size + 1, self.stride):
                    np.save(os.path.join(self.data_path, f"{os.path.splitext(filename)[0]}_{num_windows}.npy"), data[start:start + self.window_size])
                    num_windows += 1
        return os.listdir(self.data_path)
