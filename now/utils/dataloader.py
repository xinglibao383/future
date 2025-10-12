from torch.utils.data import DataLoader, random_split
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from collections import Counter


def get_major_label(start, end, label_array):
    overlap_counts = []
    for row in label_array:
        class_id = int(row[1])
        seg_start = int(row[2])
        seg_end = int(row[3])

        for _ in range(seg_start, seg_end):
            overlap_counts.append(class_id)
        
    overlap_counts = overlap_counts[start:end]
    counter = Counter(overlap_counts)
    most_common_num, _ = counter.most_common(1)[0]
    return most_common_num


def split_and_save_full_h5_segment(h5_path, window_size, stride, save_dir):
    basename = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]     # shape: (5, 2950, 6)
        label = f['label'][:]   # shape: (8, 4)

    T = data.shape[1]  # 时间长度（2950）
    num_saved = 0

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        segment = data[:, start:end, :]  # shape: (5, window_size, 6)
        major_label = get_major_label(start, end, label)
        np.save(os.path.join(save_dir, f"{basename}_{num_saved}_{major_label}.npy"), segment)
        num_saved += 1

    print(f"Processed {h5_path}, saved {num_saved} segments to {save_dir}")


def process_all_h5_in_folder(folder_path, window_size, stride, save_dir):
    save_dir = os.path.join(save_dir, f"imu_{window_size}_{stride}")
    os.makedirs(save_dir, exist_ok=True)

    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    if not h5_files:
        return

    for h5_filename in h5_files:
        h5_path = os.path.join(folder_path, h5_filename)
        split_and_save_full_h5_segment(h5_path, window_size, stride, save_dir)


class XRF55(Dataset):
    def __init__(self, root_path, window_size, stride):
        self.root_path = root_path
        self.window_size = window_size
        self.stride = stride
        self.data_path = os.path.join(self.root_path, f"imu_{self.window_size}_{self.stride}")
        self.filenames = self.get_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(os.path.join(self.data_path, self.filenames[idx])), dtype=torch.float32)
        data = data.permute(0, 2, 1).reshape(5 * 6, self.window_size)
        lable = int(os.path.splitext(os.path.basename(self.filenames[idx]))[0].split('_')[-1])
        return data, lable
    
    def get_filenames(self):
        if not os.path.isdir(self.data_path):
            process_all_h5_in_folder(
                folder_path="/data/luohonglin/imu",
                window_size=self.window_size,
                stride=self.stride,
                save_dir=self.root_path
            )
        return os.listdir(self.data_path)
    

def get_dataloaders(root_path, window_size, stride, batch_size, train_ratio):
    dataset= XRF55(root_path, window_size, stride)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))
    

class XRF55CrossPerson(Dataset):
    def __init__(self, root_path, window_size, stride, person_ids):
        self.root_path = root_path
        self.window_size = window_size
        self.stride = stride
        self.person_ids = person_ids
        self.data_path = os.path.join(self.root_path, f"imu_{self.window_size}_{self.stride}")
        self.filenames = self.get_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(os.path.join(self.data_path, self.filenames[idx])), dtype=torch.float32)
        data = data.permute(0, 2, 1).reshape(5 * 6, self.window_size)
        lable = int(os.path.splitext(os.path.basename(self.filenames[idx]))[0].split('_')[-1])
        return data, lable
    
    def get_filenames(self):
        if not os.path.isdir(self.data_path):
            process_all_h5_in_folder(
                folder_path="/data/luohonglin/imu",
                window_size=self.window_size,
                stride=self.stride,
                save_dir=self.root_path
            )
        # 将每个filename按照下划线切分，然后取第一个元素，如果这个元素在person_ids中，则保留，否则删除
        return [filename for filename in os.listdir(self.data_path) if int(filename.split('_')[0]) in self.person_ids]


def get_dataloaders_cross_person(root_path, window_size, stride, batch_size, train_person_ids, val_person_ids):
    train_dataset= XRF55CrossPerson(root_path, window_size, stride, train_person_ids)
    val_dataset= XRF55CrossPerson(root_path, window_size, stride, val_person_ids)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


class XRF55CrossEnvironment(Dataset):
    def __init__(self, root_path, window_size, stride, environment_ids):
        self.root_path = root_path
        self.window_size = window_size
        self.stride = stride
        self.environment_ids = environment_ids
        self.data_path = os.path.join(self.root_path, f"imu_{self.window_size}_{self.stride}")
        self.filenames = self.get_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = torch.tensor(np.load(os.path.join(self.data_path, self.filenames[idx])), dtype=torch.float32)
        data = data.permute(0, 2, 1).reshape(5 * 6, self.window_size)
        lable = int(os.path.splitext(os.path.basename(self.filenames[idx]))[0].split('_')[-1])
        return data, lable
    
    def get_filenames(self):
        if not os.path.isdir(self.data_path):
            process_all_h5_in_folder(
                folder_path="/data/luohonglin/imu",
                window_size=self.window_size,
                stride=self.stride,
                save_dir=self.root_path
            )
        # 将每个filename按照下划线切分，然后取第二个元素，如果这个元素在person_ids中，则保留，否则删除
        return [filename for filename in os.listdir(self.data_path) if int(filename.split('_')[1]) in self.environment_ids]


def get_dataloaders_cross_environment(root_path, window_size, stride, batch_size, train_environment_ids, val_environment_ids):
    train_dataset= XRF55CrossEnvironment(root_path, window_size, stride, train_environment_ids)
    val_dataset= XRF55CrossEnvironment(root_path, window_size, stride, val_environment_ids)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=16, persistent_workers=True), 
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=16, persistent_workers=True))


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders("/home/xinglibao/workspace/future/now/data/imu", 150, 50, 32, 0.8)
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)