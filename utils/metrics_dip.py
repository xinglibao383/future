import math
import numpy as np
import pandas as pd
import torch


DIP_IMU_24_JOINT_NAMES = [f"joint_{i}" for i in range(24)]
SMPL_POS_24_JOINT_NAMES = [f"joint_{i}" for i in range(24)]


def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(theta, min=eps)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)

    k = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).reshape(*axis.shape[:-1], 3, 3)

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis.shape[:-1], 3, 3)

    theta_expanded = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta_expanded)
    cos_theta = torch.cos(theta_expanded)

    rot = eye + sin_theta * k + (1.0 - cos_theta) * (k @ k)

    small = (theta < 1e-6).unsqueeze(-1)
    rot = torch.where(small, eye, rot)
    return rot


def rotation_angle_deg(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    rel_rot = pred_rot @ gt_rot.transpose(-1, -2)
    trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
    cos_val = (trace - 1.0) / 2.0
    cos_val = torch.clamp(cos_val, -1.0, 1.0)
    angle_rad = torch.acos(cos_val)
    return angle_rad * 180.0 / math.pi


class DIPMetricTracker:
    def __init__(self, angle_joint_names=None, pos_joint_names=None):
        self.angle_joint_names = list(angle_joint_names or DIP_IMU_24_JOINT_NAMES)
        self.pos_joint_names = list(pos_joint_names or SMPL_POS_24_JOINT_NAMES)
        self.reset()

    def reset(self):
        self.angle_sum = 0.0
        self.angle_count = 0
        self.per_joint_angle_sum = None
        self.per_joint_angle_count = 0

        self.pos_sum = 0.0
        self.pos_count = 0
        self.per_joint_pos_sum = None
        self.per_joint_pos_count = 0

    def update_angle(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor):
        pred_rot = axis_angle_to_matrix(pred_pose)
        gt_rot = axis_angle_to_matrix(gt_pose)

        angle_err = rotation_angle_deg(pred_rot, gt_rot)  # [B, T, 24]

        per_joint_sum = angle_err.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        if self.per_joint_angle_sum is None:
            self.per_joint_angle_sum = torch.zeros_like(per_joint_sum)

        self.per_joint_angle_sum += per_joint_sum
        self.per_joint_angle_count += angle_err.shape[0] * angle_err.shape[1]

        self.angle_sum += angle_err.sum().item()
        self.angle_count += angle_err.numel()

    def update_pos(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor):
        pos_err_cm = torch.norm(pred_joints - gt_joints, dim=-1) * 100.0  # [B, T, J]

        per_joint_sum = pos_err_cm.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        if self.per_joint_pos_sum is None:
            self.per_joint_pos_sum = torch.zeros_like(per_joint_sum)

        self.per_joint_pos_sum += per_joint_sum
        self.per_joint_pos_count += pos_err_cm.shape[0] * pos_err_cm.shape[1]

        self.pos_sum += pos_err_cm.sum().item()
        self.pos_count += pos_err_cm.numel()

    def summary(self):
        metrics = {}

        metrics["ang_err_deg"] = (
            self.angle_sum / self.angle_count if self.angle_count > 0 else 0.0
        )

        if self.per_joint_angle_sum is None:
            metrics["per_joint_ang_err_deg"] = np.array([], dtype=np.float64)
        else:
            metrics["per_joint_ang_err_deg"] = (
                self.per_joint_angle_sum / max(self.per_joint_angle_count, 1)
            ).numpy()

        if self.pos_count > 0:
            metrics["pos_err_cm"] = self.pos_sum / self.pos_count
            metrics["per_joint_pos_err_cm"] = (
                self.per_joint_pos_sum / max(self.per_joint_pos_count, 1)
            ).numpy()
        else:
            metrics["pos_err_cm"] = None
            metrics["per_joint_pos_err_cm"] = None

        return metrics

    def format_angle_metrics(self, metrics, label="mean joint angle error"):
        return f"{label}: {metrics['ang_err_deg']:.4f} deg"

    def format_pos_metrics(self, metrics, label="positional error"):
        if metrics["pos_err_cm"] is None:
            return f"{label}: N/A"
        return f"{label}: {metrics['pos_err_cm']:.4f} cm"

    def format_per_joint_angle(self, per_joint_angle):
        joint_count = len(per_joint_angle)
        joint_names = (
            self.angle_joint_names[:joint_count]
            if joint_count <= len(self.angle_joint_names)
            else [f"joint_{i}" for i in range(joint_count)]
        )
        df = pd.DataFrame({
            "joint_id": np.arange(joint_count),
            "joint_name": joint_names,
            "ang_err_deg": np.round(per_joint_angle, 4),
        })
        return df.to_string(index=False)

    def format_per_joint_pos(self, per_joint_pos):
        joint_count = len(per_joint_pos)
        joint_names = (
            self.pos_joint_names[:joint_count]
            if joint_count <= len(self.pos_joint_names)
            else [f"joint_{i}" for i in range(joint_count)]
        )
        df = pd.DataFrame({
            "joint_id": np.arange(joint_count),
            "joint_name": joint_names,
            "pos_err_cm": np.round(per_joint_pos, 4),
        })
        return df.to_string(index=False)