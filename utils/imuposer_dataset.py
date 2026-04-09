import os
import shutil
import pickle
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_all_pkl_files(data_root: str) -> List[str]:
    pkl_files = []
    for current_root, dirnames, filenames in os.walk(data_root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if filename.lower().endswith(".pkl"):
                pkl_files.append(os.path.join(current_root, filename))
    return pkl_files


def process_imuposer(
    source_dir: str,
    target_dir: str,
    use_len: float,
    compute_len: float,
    predict_len: float,
    stride_len: float,
    fps: int = 25,
):
    """
    将 IMUPoser 原始 pkl 数据切成样本，保存为:
        target_dir/
            imu/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy
            pose/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy

    这里 use_len / compute_len / predict_len / stride_len 单位均为“秒”
    内部自动转换为帧数:
        帧数 = 秒数 * fps
    """

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    imu_target_dir = os.path.join(
        target_dir, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}"
    )
    pose_target_dir = os.path.join(
        target_dir, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}"
    )

    if os.path.exists(imu_target_dir):
        shutil.rmtree(imu_target_dir)
    if os.path.exists(pose_target_dir):
        shutil.rmtree(pose_target_dir)

    os.makedirs(imu_target_dir, exist_ok=True)
    os.makedirs(pose_target_dir, exist_ok=True)

    use_len_frames = int(use_len * fps)
    compute_len_frames = int(compute_len * fps)
    predict_len_frames = int(predict_len * fps)
    stride_len_frames = int(stride_len * fps)

    pkl_files = collect_all_pkl_files(source_dir)

    total_seq_count = 0
    total_sample_count = 0

    imu_total_len = use_len_frames + predict_len_frames
    pose_total_len = compute_len_frames + predict_len_frames

    for pkl_path in pkl_files:
        total_seq_count += 1
        print(f"Processing: {pkl_path}")

        try:
            data = load_pickle_file(pkl_path)
        except Exception as e:
            print(f"[Skip] failed to load {pkl_path}: {repr(e)}")
            continue

        if not isinstance(data, dict):
            print(f"[Skip] file is not dict: {pkl_path}")
            continue

        if "imu" not in data or "pose" not in data:
            print(f"[Skip] missing 'imu' or 'pose': {pkl_path}")
            continue

        imu = data["imu"]
        pose = data["pose"]

        if isinstance(imu, torch.Tensor):
            imu = imu.detach().cpu().numpy()
        else:
            imu = np.asarray(imu)

        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        else:
            pose = np.asarray(pose)

        imu = imu.astype(np.float32)    # (T, 60)
        pose = pose.astype(np.float32)  # (T, 72)

        if imu.ndim != 2 or imu.shape[1] != 60:
            print(f"[Skip] unexpected imu shape {imu.shape}: {pkl_path}")
            continue

        if pose.ndim != 2 or pose.shape[1] != 72:
            print(f"[Skip] unexpected pose shape {pose.shape}: {pkl_path}")
            continue

        seq_len = min(len(imu), len(pose))
        imu = imu[:seq_len]
        pose = pose[:seq_len]

        if seq_len < max(imu_total_len, pose_total_len):
            print(f"[Skip] sequence too short ({seq_len}): {pkl_path}")
            continue

        pose = pose.reshape(seq_len, 24, 3)  # (T, 24, 3)

        max_start = seq_len - max(imu_total_len, pose_total_len)

        rel_path = os.path.relpath(pkl_path, source_dir)
        rel_stem = rel_path.replace(os.sep, "__").replace(".pkl", "")

        for start_idx in range(0, max_start + 1, stride_len_frames):
            imu_start = start_idx
            imu_end = imu_start + imu_total_len

            pose_start = start_idx
            pose_end = pose_start + pose_total_len

            imu_sub = imu[imu_start:imu_end]      # (use+pred, 60)
            pose_sub = pose[pose_start:pose_end]  # (compute+pred, 24, 3)

            imu_sub = imu_sub.transpose(1, 0)     # (60, use+pred)

            save_name = f"{rel_stem}_{start_idx}.npy"

            np.save(
                os.path.join(imu_target_dir, save_name),
                imu_sub.astype(np.float32)
            )
            np.save(
                os.path.join(pose_target_dir, save_name),
                pose_sub.astype(np.float32)
            )

            total_sample_count += 1

    print("\n处理完成")
    print(f"原始序列数: {total_seq_count}")
    print(f"总样本数: {total_sample_count}")
    print(f"IMU 保存目录: {imu_target_dir}")
    print(f"Pose 保存目录: {pose_target_dir}")


class IMUPoserDataset(Dataset):
    def __init__(
        self,
        use_len: float,
        compute_len: float,
        predict_len: float,
        stride_len: float,
        fps: int = 25,
    ):
        """
        参数单位均为“秒”
        """
        super().__init__()

        root_path = "/mnt/mydata/yh/liming/workspace/future/mydata/IMUPOSER_split"
        source_dir = "/mnt/mydata/yh/liming/workspace/imuposer_dataset"

        self.root_path = root_path
        self.use_len = use_len
        self.compute_len = compute_len
        self.predict_len = predict_len
        self.stride_len = stride_len
        self.fps = fps

        self.use_len_frames = int(use_len * fps)
        self.compute_len_frames = int(compute_len * fps)
        self.predict_len_frames = int(predict_len * fps)
        self.stride_len_frames = int(stride_len * fps)

        self.imu_root_path = os.path.join(
            root_path, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}"
        )
        self.pose_root_path = os.path.join(
            root_path, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}"
        )

        if (not os.path.exists(self.imu_root_path)) or (not os.path.exists(self.pose_root_path)):
            process_imuposer(
                source_dir=source_dir,
                target_dir=root_path,
                use_len=use_len,
                compute_len=compute_len,
                predict_len=predict_len,
                stride_len=stride_len,
                fps=fps,
            )

        filenames = sorted(os.listdir(self.imu_root_path))
        self.imu_filepaths = [os.path.join(self.imu_root_path, f) for f in filenames]
        self.pose_filepaths = [os.path.join(self.pose_root_path, f) for f in filenames]

        self.cache = {}

    def __len__(self):
        return len(self.imu_filepaths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32)

        imu = torch.nan_to_num(imu, nan=0.0, posinf=0.0, neginf=0.0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)

        hist_imu = imu[:, :self.use_len_frames]
        current_pose = pose[:self.compute_len_frames]
        future_imu = imu[:, self.use_len_frames:]
        future_pose = pose[self.compute_len_frames:]

        self.cache[idx] = (hist_imu, current_pose, future_imu, future_pose)
        return self.cache[idx]


if __name__ == "__main__":
    dataset = IMUPoserDataset(
        use_len=4,
        compute_len=1,
        predict_len=1,
        stride_len=1,
        fps=25,
    )

    print("数据集大小:", len(dataset))
    hist_imu, current_pose, future_imu, future_pose = dataset[0]
    print("历史 IMU shape:", hist_imu.shape)
    print("当前姿态 shape:", current_pose.shape)
    print("未来 IMU shape:", future_imu.shape)
    print("未来姿态 shape:", future_pose.shape)