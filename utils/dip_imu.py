import os
import shutil
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def process(source_dir, target_dir, use_len, compute_len, predict_len, stride_len, fps=60):
    """
    将 DIP-IMU 原始 pkl 数据切成样本，保存为:
        target_dir/
            imu/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy
            pose/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy

    这里的 use_len / compute_len / predict_len / stride_len 单位都是“秒”。
    内部会自动转换为帧数:
        帧数 = 秒数 * fps
    """

    data_root = os.path.join(source_dir, "DIP_IMU")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"DIP_IMU 目录不存在: {data_root}")

    imu_target_dir = os.path.join(target_dir, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
    pose_target_dir = os.path.join(target_dir, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")

    if os.path.exists(imu_target_dir):
        shutil.rmtree(imu_target_dir)
    if os.path.exists(pose_target_dir):
        shutil.rmtree(pose_target_dir)

    os.makedirs(imu_target_dir, exist_ok=True)
    os.makedirs(pose_target_dir, exist_ok=True)

    # 秒 -> 帧
    use_len_frames = int(use_len * fps)
    compute_len_frames = int(compute_len * fps)
    predict_len_frames = int(predict_len * fps)
    stride_len_frames = int(stride_len * fps)

    total_count = 0
    total_seq_count = 0

    imu_total_len = use_len_frames + predict_len_frames
    pose_total_len = compute_len_frames + predict_len_frames

    for subject in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject)
        if not os.path.isdir(subject_dir):
            continue

        for fname in sorted(os.listdir(subject_dir)):
            if not fname.endswith(".pkl"):
                continue

            filepath = os.path.join(subject_dir, fname)
            print(f"Processing: {filepath}")
            total_seq_count += 1

            with open(filepath, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            gt = np.asarray(data["gt"], dtype=np.float32)              # (T, 72)
            imu_acc = np.asarray(data["imu_acc"], dtype=np.float32)    # (T, 17, 3)
            imu_ori = np.asarray(data["imu_ori"], dtype=np.float32)    # (T, 17, 3, 3)

            imu_ori_flat = imu_ori.reshape(imu_ori.shape[0], imu_ori.shape[1], -1)   # (T, 17, 9)
            imu = np.concatenate([imu_ori_flat, imu_acc], axis=-1)                    # (T, 17, 12)

            seq_len = min(gt.shape[0], imu.shape[0])
            gt = gt[:seq_len]
            imu = imu[:seq_len]

            max_start = seq_len - max(imu_total_len, pose_total_len)
            if max_start < 0:
                continue

            for start_idx in range(0, max_start + 1, stride_len_frames):
                imu_start_idx = start_idx
                imu_end_idx = imu_start_idx + imu_total_len

                pose_start_idx = start_idx
                pose_end_idx = pose_start_idx + pose_total_len

                imu_sub_data = imu[imu_start_idx:imu_end_idx]      # (use_len+predict_len, 17, 12)
                pose_sub_data = gt[pose_start_idx:pose_end_idx]    # (compute_len+predict_len, 72)
                pose_sub_data = pose_sub_data.reshape(pose_sub_data.shape[0], 24, 3)  # -> (T, 24, 3)

                imu_sub_data = np.transpose(imu_sub_data, (1, 2, 0)).reshape(-1, imu_sub_data.shape[0])

                save_name = f"{subject}_{fname.replace('.pkl', '')}_{start_idx}.npy"
                np.save(os.path.join(imu_target_dir, save_name), imu_sub_data.astype(np.float32))
                np.save(os.path.join(pose_target_dir, save_name), pose_sub_data.astype(np.float32))

                total_count += 1

    print(f"\n处理完成")
    print(f"原始序列数: {total_seq_count}")
    print(f"总样本数: {total_count}")
    print(f"IMU 保存目录: {imu_target_dir}")
    print(f"Pose 保存目录: {pose_target_dir}")


class DIP_IMU(Dataset):
    def __init__(self, use_len, compute_len, predict_len, stride_len, fps=60):
        """
        这里的 use_len / compute_len / predict_len / stride_len 单位都是“秒”
        """
        super().__init__()
        root_path = '/mnt/mydata/yh/liming/workspace/future/mydata/DIP_IMU_split'
        self.imu_root_path = os.path.join(root_path, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
        self.pose_root_path = os.path.join(root_path, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")

        self.use_len = use_len
        self.compute_len = compute_len
        self.predict_len = predict_len
        self.stride_len = stride_len
        self.fps = fps

        # 秒 -> 帧
        self.use_len_frames = int(use_len * fps)
        self.compute_len_frames = int(compute_len * fps)
        self.predict_len_frames = int(predict_len * fps)
        self.stride_len_frames = int(stride_len * fps)

        if not os.path.exists(self.imu_root_path) or not os.path.exists(self.pose_root_path):
            process(
                source_dir="/mnt/mydata/yh/liming/data/DIP_IMU_and_Others",
                target_dir=root_path,
                use_len=use_len,
                compute_len=compute_len,
                predict_len=predict_len,
                stride_len=stride_len,
                fps=fps
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

        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)   # (204, use_len_frames+predict_len_frames)
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32) # (compute_len_frames+predict_len_frames, 24, 3)
        imu = torch.nan_to_num(imu, nan=0.0, posinf=0.0, neginf=0.0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        hist_imu = imu[:, :self.use_len_frames]
        current_pose = pose[:self.compute_len_frames]
        future_imu = imu[:, self.use_len_frames:]
        future_pose = pose[self.compute_len_frames:]

        self.cache[idx] = (hist_imu, current_pose, future_imu, future_pose)
        return self.cache[idx]


if __name__ == "__main__":
    dataset = DIP_IMU(
        root_path="/mnt/mydata/yh/liming/workspace/future/mydata/DIP_IMU_split",
        use_len=4,
        compute_len=1,
        predict_len=1,
        stride_len=1,
        fps=60
    )

    print("数据集大小:", len(dataset))
    hist_imu, current_pose, future_imu, future_pose = dataset[0]
    print("历史 IMU shape:", hist_imu.shape)
    print("当前姿态 shape:", current_pose.shape)
    print("未来 IMU shape:", future_imu.shape)
    print("未来姿态 shape:", future_pose.shape)