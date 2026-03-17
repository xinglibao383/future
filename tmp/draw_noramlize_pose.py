import os
import shutil
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


class PoseNormalizationVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # ===== 完全使用你提供的骨架 =====
        self.skeleton = [
            (0,1),(1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),
            (1,8),(8,9),(9,10),(10,11),
            (8,12),(12,13),(13,14),
            (0,15),(15,17),(0,16),(16,18),
            (14,19),(19,20),(14,21),
            (11,22),(22,23),(11,24)
        ]

        # 与你Dataset一致
        self.center_idx = 8
        self.left_shoulder_idx = 5
        self.right_shoulder_idx = 2

    # ========================
    # 数据处理（严格复现Dataset）
    # ========================
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

    def normalize_pose(self, keypoints_tensor, center_idx=8, left_shoulder_idx=5, right_shoulder_idx=2):
        center = keypoints_tensor[:, center_idx, :2].unsqueeze(1)  # (num_poses, 1, 2)
        keypoints_centered = keypoints_tensor.clone()
        keypoints_centered[:, :, :2] -= center
        l_shoulder = keypoints_centered[:, left_shoulder_idx, :2]  # (num_poses, 2)
        r_shoulder = keypoints_centered[:, right_shoulder_idx, :2]
        shoulder_width = torch.norm(l_shoulder - r_shoulder, dim=1).unsqueeze(1).unsqueeze(2)  # (num_poses,1,1)
        shoulder_width_clamped = torch.clamp(shoulder_width, min=1e-6)  # 防止除零
        keypoints_centered[:, :, :2] /= shoulder_width_clamped
        keypoints_centered[:, :, :2] = torch.tanh(keypoints_centered[:, :, :2])
        return keypoints_centered

    # ========================
    # 画图（严格风格统一）
    # ========================
    def _draw_pose(self, ax, pose):
        x = pose[:, 0].numpy()
        y = pose[:, 1].numpy()

        ax.scatter(x, y, c='red', s=20)

        for i, j in self.skeleton:
            ax.plot([x[i], x[j]], [y[i], y[j]], 'g-', linewidth=2)

        ax.invert_yaxis()
        ax.axis('equal')
        ax.axis('off')

    # ========================
    # 核心函数
    # ========================
    def visualize(self, npy_path, max_frames=2):
        filename = os.path.basename(npy_path).replace(".npy", "")
        poses = torch.tensor(np.load(npy_path), dtype=torch.float32)

        # ===== Step1: 补点 =====
        poses_filled = self.fill_missing_keypoints(poses)

        # ===== Step2: 归一化 =====
        poses_norm = self.normalize_pose(poses_filled)

        num_frames = min(max_frames, poses.shape[0])

        fig, axes = plt.subplots(num_frames, 2, figsize=(6, 3 * num_frames))

        for i in range(num_frames):
            raw = poses_filled[i, :, :2]
            norm = poses_norm[i, :, :2]

            self._draw_pose(axes[i, 0], raw)
            axes[i, 0].set_title("Before")

            self._draw_pose(axes[i, 1], norm)
            axes[i, 1].set_title("After")

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"[Saved] {save_path}")


# ========================
# 使用方式
# ========================
if __name__ == "__main__":
    pose_dir = "/mnt/mydata/yh/liming/workspace/future/mydata/pose/60_15_15_15"
    save_dir = "/mnt/mydata/yh/liming/workspace/future/tmp/imgs"

    visualizer = PoseNormalizationVisualizer(save_dir)

    file_list = os.listdir(pose_dir)
    random.shuffle(file_list)

    for v in file_list:
        target_file = os.path.join(pose_dir, v)
        visualizer.visualize(target_file)