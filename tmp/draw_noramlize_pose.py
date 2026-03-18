import os
import shutil
from tqdm import tqdm
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class PoseNormalizationVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.skeleton = [
            (0,1),(1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),
            (1,8),(8,9),(9,10),(10,11),
            (8,12),(12,13),(13,14),
            (0,15),(15,17),(0,16),(16,18),
            (14,19),(19,20),(14,21),
            (11,22),(22,23),(11,24)
        ]

        self.center_idx = 8
        self.left_shoulder_idx = 5
        self.right_shoulder_idx = 2

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
            if poses[-1][j][2] == 0:
                for k in range(num_poses - 2, -1, -1):
                    if poses[k][j][2] != 0:
                        poses[-1][j] = poses[k][j]
                        break

        return poses

    # ========================
    def normalize_pose(self, keypoints_tensor):
        center = keypoints_tensor[:, self.center_idx, :2].unsqueeze(1)

        centered = keypoints_tensor.clone()
        centered[:, :, :2] -= center

        l = centered[:, self.left_shoulder_idx, :2]
        r = centered[:, self.right_shoulder_idx, :2]

        shoulder_width = torch.norm(l - r, dim=1).view(-1,1,1)
        shoulder_width = torch.clamp(shoulder_width, min=1e-3)  # ⭐防止爆炸

        centered[:, :, :2] /= shoulder_width
        centered[:, :, :2] = torch.tanh(centered[:, :, :2])

        return centered

    # ========================
    def compute_processed_range(self, pose):
        pts = pose.reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return (-3, 3, -3, 3)
        max_x = np.max(np.abs(pts[:, 0]))
        max_y = np.max(np.abs(pts[:, 1]))
        max_range = max(max_x, max_y)
        max_range = max(max_range, 3.0)
        max_range *= 1.1  # padding
        return (-max_range, max_range, -max_range, max_range)

    # ========================
    def _draw_pose(self, ax, pose, coord_range, title):
        x = pose[:, 0].numpy()
        y = pose[:, 1].numpy()

        ax.scatter(x, y, s=15)

        for i, j in self.skeleton:
            ax.plot([x[i], x[j]], [y[i], y[j]], linewidth=1.5)

        xmin, xmax, ymin, ymax = coord_range

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")

        # ⭐关键：刻度字体大小
        ax.tick_params(axis='both', labelsize=10)

        # ax.grid(True, linestyle="--", alpha=0.3)

    # ========================
    def visualize(self, npy_path):
        filename = os.path.basename(npy_path).replace(".npy", "")
        poses = torch.tensor(np.load(npy_path), dtype=torch.float32)

        poses_filled = self.fill_missing_keypoints(poses)
        poses_norm = self.normalize_pose(poses_filled)

        # ⭐关键：atanh恢复结构
        # poses_vis = torch.atanh(poses_norm.clamp(-0.9999, 0.9999))
        poses_vis = torch.atanh(poses_norm)

        for i in range(poses.shape[0]):
            if i > 0:
                break
            raw = poses_filled[i, :, :2]
            vis = poses_vis[i, :, :2]

            norm_range = self.compute_processed_range(vis.numpy())

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))

            self._draw_pose(axes[0], raw, (-1500, 1500, -1500, 1500), "Raw")
            self._draw_pose(axes[1], vis, norm_range, "Centered And Normalized")

            save_path = os.path.join(self.save_dir, f"{filename}_frame_{i}.png")

            plt.tight_layout()
            plt.savefig(save_path, dpi=600)
            plt.close()


# ========================
if __name__ == "__main__":
    pose_dir = "/mnt/mydata/yh/liming/workspace/future/mydata/pose/60_15_15_15"
    save_dir = f"/mnt/mydata/yh/liming/workspace/future/tmp/imgs"
    # save_dir = f"/mnt/mydata/yh/liming/workspace/future/tmp/imgs/{datetime.now().strftime('%Y%m%d%H%M%S')}"

    visualizer = PoseNormalizationVisualizer(save_dir)

    file_list = os.listdir(pose_dir)
    # random.shuffle(file_list)

    for v in tqdm(file_list):
        visualizer.visualize(os.path.join(pose_dir, v))