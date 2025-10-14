import os
import torch
import numpy as np
from torch.utils.data import Dataset


class XRFV22(Dataset):
    def __init__(self, root_path, use_len, compute_len, predict_len, stride_len):
        super().__init__()
        self.root_path = root_path
        self.use_len = use_len
        self.compute_len = compute_len
        self.predict_len = predict_len
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
        if imu.shape[1] == 0:
            imu = torch.zeros((30, int((self.use_len + self.predict_len) / 15 * 50)))
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32)
        normalized_pose = self.fill_missing_keypoints(self.normalize_pose(pose))
        # return imu[:, :int(self.use_len / 15 * 50)], pose[:self.compute_len, :, :2], imu[:, int(self.use_len / 15 * 50):], pose[self.compute_len:, :, :2]
        return imu[:, :int(self.use_len / 15 * 50)], normalized_pose[:self.compute_len, :, :2], imu[:, int(self.use_len / 15 * 50):], normalized_pose[self.compute_len:, :, :2]

    def normalize_pose(self, keypoints_tensor, center_idx=8, left_shoulder_idx=5, right_shoulder_idx=2):
        center = keypoints_tensor[:, center_idx, :2].unsqueeze(1)  # (num_poses, 1, 2)
        keypoints_centered = keypoints_tensor.clone()
        keypoints_centered[:, :, :2] -= center
        l_shoulder = keypoints_centered[:, left_shoulder_idx, :2]  # (num_poses, 2)
        r_shoulder = keypoints_centered[:, right_shoulder_idx, :2]
        shoulder_width = torch.norm(l_shoulder - r_shoulder, dim=1).unsqueeze(1).unsqueeze(2)  # (num_poses,1,1)
        shoulder_width = torch.clamp(shoulder_width, min=1e-6)  # 防止除零
        keypoints_centered[:, :, :2] /= shoulder_width
        # 新加的
        keypoints_centered[:, :, :2] = torch.tanh(keypoints_centered[:, :, :2])
        # print(f"最大值: {keypoints_centered.max().item()}, 最小值: {keypoints_centered.min().item()}")
        return keypoints_centered

    def fill_missing_keypoints(self, poses, num_keypoints=25):
        num_poses = poses.shape[0]
        for i in range(num_poses - 1):
            for j in range(num_keypoints):
                if poses[i][j][2] == 0:
                    for k in range(i + 1, num_poses):
                        if poses[k][j][2] != 0:
                            poses[i][j] = poses[k][j]
                            break
        for j in range(num_keypoints):
            if poses[num_poses - 1][j][2] == 0:
                for k in range(num_poses - 2, -1, -1):
                    if poses[k][j][2] != 0:
                        poses[num_poses - 1][j] = poses[k][j]
                        break
        return poses