import numpy as np
import pandas as pd
import torch


DEFAULT_PCK_THRESHOLD_RATIOS = [0.05, 0.10, 0.20]
DEFAULT_PCK_THRESHOLD_PIXELS = [5.0, 10.0, 20.0]

DIP_IMU_24_JOINT_NAMES = [
    "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "10", "11",
    "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "23",
]


def ratio_to_key(ratio):
    return f"pck@{ratio:.2f}"


def pixel_to_key(pixel):
    if float(pixel).is_integer():
        return f"pck@{int(pixel)}px"
    return f"pck@{pixel:g}px"


class PoseMetricTracker:
    def __init__(
        self,
        prefixes=None,
        pck_threshold_ratios=None,
        pck_threshold_pixels=None,
        joint_names=None,
        enable_ratio_pck=False,
        enable_pixel_pck=False,
    ):
        self.prefixes = list(prefixes) if prefixes is not None else [None]
        self.pck_threshold_ratios = list(pck_threshold_ratios or DEFAULT_PCK_THRESHOLD_RATIOS)
        self.pck_threshold_pixels = list(pck_threshold_pixels or DEFAULT_PCK_THRESHOLD_PIXELS)
        self.joint_names = list(joint_names or DIP_IMU_24_JOINT_NAMES)
        self.enable_ratio_pck = enable_ratio_pck
        self.enable_pixel_pck = enable_pixel_pck
        self.reset()

    def reset(self):
        self.state = {}
        for prefix in self.prefixes:
            prefix_state = {
                "error_sum": 0.0,
                "error_count": 0,
                "per_joint_error_sum": None,
                "per_joint_count": 0,
            }
            if self.enable_ratio_pck:
                for ratio in self.pck_threshold_ratios:
                    key = ratio_to_key(ratio)
                    prefix_state[f"{key}_correct"] = 0.0
                    prefix_state[f"{key}_count"] = 0
            if self.enable_pixel_pck:
                for pixel in self.pck_threshold_pixels:
                    key = pixel_to_key(pixel)
                    prefix_state[f"{key}_correct"] = 0.0
                    prefix_state[f"{key}_count"] = 0
            self.state[prefix] = prefix_state

    def update(self, pred_pose, gt_pose, shoulder_width, prefix=None):
        if prefix not in self.state:
            raise KeyError(f"Unknown metric prefix: {prefix}")

        joint_errors = torch.norm(pred_pose - gt_pose, dim=-1)

        prefix_state = self.state[prefix]
        per_joint_error_sum = joint_errors.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        if prefix_state["per_joint_error_sum"] is None:
            prefix_state["per_joint_error_sum"] = torch.zeros_like(per_joint_error_sum)
        prefix_state["per_joint_error_sum"] += per_joint_error_sum
        prefix_state["per_joint_count"] += joint_errors.shape[0] * joint_errors.shape[1]

        prefix_state["error_sum"] += joint_errors.sum().item()
        prefix_state["error_count"] += joint_errors.numel()

    def summary(self):
        summary = {}
        for prefix in self.prefixes:
            prefix_state = self.state[prefix]
            prefix_name = "" if prefix is None else f"{prefix}_"
            error_count = prefix_state["error_count"]
            summary[f"{prefix_name}mjpe"] = float(prefix_state["error_sum"] / error_count) if error_count else 0.0

            if prefix_state["per_joint_error_sum"] is None:
                per_joint_mjpe = np.array([], dtype=np.float64)
            else:
                per_joint_mjpe = (
                    prefix_state["per_joint_error_sum"] / max(prefix_state["per_joint_count"], 1)
                ).numpy()
            summary[f"{prefix_name}per_joint_mjpe"] = per_joint_mjpe
        return summary

    def format_mjpe_metrics(self, metrics, prefix=None, label=None):
        prefix_name = "" if prefix is None else f"{prefix}_"
        metric_name = f"{prefix_name}mjpe"
        show_name = label if label is not None else metric_name
        return f"{show_name}: {metrics[metric_name]:.4f}"

    def format_pose_metric_lines(self, metrics, prefix=None, mjpe_label=None, pixel_pck_label=None):
        return (self.format_mjpe_metrics(metrics, prefix=prefix, label=mjpe_label),)

    def format_per_joint_mjpe(self, per_joint_mjpe):
        joint_count = len(per_joint_mjpe)
        joint_names = (
            self.joint_names[:joint_count]
            if joint_count <= len(self.joint_names)
            else [f"{i}" for i in range(joint_count)]
        )
        df = pd.DataFrame({
            "joint_id": np.arange(joint_count),
            "joint_name": joint_names,
            "mjpe": np.round(per_joint_mjpe, 4),
        })
        return df.to_string(index=False)