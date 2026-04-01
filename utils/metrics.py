import numpy as np
import pandas as pd
import torch


DEFAULT_PCK_THRESHOLD_RATIOS = [0.05, 0.10, 0.20]
OPENPOSE_25_JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]


def ratio_to_key(ratio):
    return f"pck@{ratio:.2f}"


def restore_pose(normalized_pose, shoulder_width):
    normalized_pose = normalized_pose.clamp(min=-0.9999, max=0.9999)
    return torch.atanh(normalized_pose) * shoulder_width


class PoseMetricTracker:
    def __init__(self, prefixes=None, pck_threshold_ratios=None, joint_names=None):
        self.prefixes = list(prefixes) if prefixes is not None else [None]
        self.pck_threshold_ratios = list(pck_threshold_ratios or DEFAULT_PCK_THRESHOLD_RATIOS)
        self.joint_names = list(joint_names or OPENPOSE_25_JOINT_NAMES)
        self.reset()

    def reset(self):
        self.state = {}
        for prefix in self.prefixes:
            self.state[prefix] = {
                "error_sum": 0.0,
                "error_count": 0,
                "per_joint_error_sum": None,
                "per_joint_count": 0,
            }
            for ratio in self.pck_threshold_ratios:
                key = ratio_to_key(ratio)
                self.state[prefix][f"{key}_correct"] = 0.0
                self.state[prefix][f"{key}_count"] = 0

    def update(self, pred_pose, gt_pose, shoulder_width, prefix=None):
        if prefix not in self.state:
            raise KeyError(f"Unknown metric prefix: {prefix}")

        pred_pose = restore_pose(pred_pose, shoulder_width)
        gt_pose = restore_pose(gt_pose, shoulder_width)
        joint_errors = torch.norm(pred_pose - gt_pose, dim=-1)

        prefix_state = self.state[prefix]
        per_joint_error_sum = joint_errors.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
        if prefix_state["per_joint_error_sum"] is None:
            prefix_state["per_joint_error_sum"] = torch.zeros_like(per_joint_error_sum)
        prefix_state["per_joint_error_sum"] += per_joint_error_sum
        prefix_state["per_joint_count"] += joint_errors.shape[0] * joint_errors.shape[1]

        prefix_state["error_sum"] += joint_errors.sum().item()
        prefix_state["error_count"] += joint_errors.numel()

        base_threshold = shoulder_width.squeeze(-1).squeeze(-1)
        for ratio in self.pck_threshold_ratios:
            key = ratio_to_key(ratio)
            threshold = base_threshold * ratio
            correct = (joint_errors <= threshold.unsqueeze(-1)).float()
            prefix_state[f"{key}_correct"] += correct.sum().item()
            prefix_state[f"{key}_count"] += correct.numel()

    def summary(self):
        summary = {}
        for prefix in self.prefixes:
            prefix_state = self.state[prefix]
            prefix_name = "" if prefix is None else f"{prefix}_"
            error_count = prefix_state["error_count"]
            summary[f"{prefix_name}mpjpe"] = float(prefix_state["error_sum"] / error_count) if error_count else 0.0

            if prefix_state["per_joint_error_sum"] is None:
                per_joint_mpjpe = np.array([], dtype=np.float64)
            else:
                per_joint_mpjpe = (
                    prefix_state["per_joint_error_sum"] / max(prefix_state["per_joint_count"], 1)
                ).numpy()
            summary[f"{prefix_name}per_joint_mpjpe"] = per_joint_mpjpe

            for ratio in self.pck_threshold_ratios:
                key = ratio_to_key(ratio)
                count = prefix_state[f"{key}_count"]
                summary[f"{prefix_name}{key}"] = (
                    float(prefix_state[f"{key}_correct"] / count) if count else 0.0
                )
        return summary

    def format_pck_metrics(self, metrics, prefix=None):
        prefix_name = "" if prefix is None else f"{prefix}_"
        parts = []
        for ratio in self.pck_threshold_ratios:
            key = ratio_to_key(ratio)
            parts.append(f"{prefix_name}{key}: {metrics[f'{prefix_name}{key}']:.4f}")
        return ", ".join(parts)

    def format_per_joint_mpjpe(self, per_joint_mpjpe):
        joint_count = len(per_joint_mpjpe)
        joint_names = (
            self.joint_names[:joint_count]
            if joint_count <= len(self.joint_names)
            else [f"joint_{i}" for i in range(joint_count)]
        )
        df = pd.DataFrame({
            "joint_id": np.arange(joint_count),
            "joint_name": joint_names,
            "mpjpe": np.round(per_joint_mpjpe, 4),
        })
        return df.to_string(index=False)
