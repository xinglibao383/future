from torch.utils.data import DataLoader, random_split
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from collections import Counter


def query_label(label_data, start, end):
    ids = []
    for row in label_data:
        for _ in range(int(row[2]), int(row[3])):
            ids.append(int(row[1]))
    ids = ids[start:end]
    counter = Counter(ids)
    label, _ = counter.most_common(1)[0]
    return label


def process_raw_imu_data(data_path, save_path, window_size, stride):
    def process(filepath, save_path, window_size, stride):
        print(f"Processing: {filepath}")
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with h5py.File(filepath, 'r') as f:
            imu_data, label_data = f['data'][:], f['label'][:]
            # (5, time_len, 6), (num_segments, 4)
            # print(imu_data.shape, label_data.shape)
        for start in range(0, imu_data.shape[1] - window_size, stride):
            end = start + window_size
            imu = imu_data[:, start:end, :]             
            imu = np.transpose(imu, (0, 2, 1))  
            imu = imu.reshape(5 * 6, window_size)
            label = query_label(label_data, start, end)
            np.save(os.path.join(save_path, f"{basename}_{start}_{end}_{label}.npy"), imu)
    save_path = os.path.join(save_path, f"{window_size}_{stride}")
    os.makedirs(save_path, exist_ok=True)
    for filename in os.listdir(data_path):
        process(os.path.join(data_path, filename), save_path, window_size, stride)


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/utils/data_process4_c.py
if __name__ == "__main__":
    process_raw_imu_data(
        data_path="/mnt/mydata/yh/liming/data/xrfv2/imu", 
        save_path="/mnt/mydata/yh/liming/workspace/future/mydatac/imu", 
        window_size=15, 
        stride=15
    )