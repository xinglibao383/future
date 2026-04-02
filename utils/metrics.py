import numpy as np
import pandas as pd
import torch


DEFAULT_PCK_THRESHOLD_RATIOS = [0.05, 0.10, 0.20]
DEFAULT_PCK_THRESHOLD_PIXELS = [5.0, 10.0, 20.0]
OPENPOSE_25_JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]


def ratio_to_key(ratio):
    return f"pck@{ratio:.2f}"


def pixel_to_key(pixel):
    if float(pixel).is_integer():
        return f"pck@{int(pixel)}px"
    return f"pck@{pixel:g}px"


def restore_pose(normalized_pose, shoulder_width):
    normalized_pose = normalized_pose.clamp(min=-0.9999, max=0.9999)
    return torch.atanh(normalized_pose) * shoulder_width


class PoseMetricTracker:
    def __init__(
        self,
        prefixes=None,
        pck_threshold_ratios=None,
        pck_threshold_pixels=None,
        joint_names=None,
        enable_ratio_pck=False,
        enable_pixel_pck=True,
    ):
        self.prefixes = list(prefixes) if prefixes is not None else [None]
        self.pck_threshold_ratios = list(pck_threshold_ratios or DEFAULT_PCK_THRESHOLD_RATIOS)
        self.pck_threshold_pixels = list(pck_threshold_pixels or DEFAULT_PCK_THRESHOLD_PIXELS)
        self.joint_names = list(joint_names or OPENPOSE_25_JOINT_NAMES)
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

        if self.enable_ratio_pck:
            base_threshold = shoulder_width.squeeze(-1).squeeze(-1)
            for ratio in self.pck_threshold_ratios:
                key = ratio_to_key(ratio)
                threshold = base_threshold * ratio
                correct = (joint_errors <= threshold.unsqueeze(-1)).float()
                prefix_state[f"{key}_correct"] += correct.sum().item()
                prefix_state[f"{key}_count"] += correct.numel()

        if self.enable_pixel_pck:
            for pixel in self.pck_threshold_pixels:
                key = pixel_to_key(pixel)
                correct = (joint_errors <= float(pixel)).float()
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

            if self.enable_ratio_pck:
                for ratio in self.pck_threshold_ratios:
                    key = ratio_to_key(ratio)
                    count = prefix_state[f"{key}_count"]
                    summary[f"{prefix_name}{key}"] = (
                        float(prefix_state[f"{key}_correct"] / count) if count else 0.0
                    )

            if self.enable_pixel_pck:
                for pixel in self.pck_threshold_pixels:
                    key = pixel_to_key(pixel)
                    count = prefix_state[f"{key}_count"]
                    summary[f"{prefix_name}{key}"] = (
                        float(prefix_state[f"{key}_correct"] / count) if count else 0.0
                    )
        return summary

    def format_mpjpe_metrics(self, metrics, prefix=None, label=None):
        prefix_name = "" if prefix is None else f"{prefix}_"
        metric_name = f"{prefix_name}mpjpe"
        show_name = label if label is not None else metric_name
        return f"{show_name}: {metrics[metric_name]:.4f}"

    def _format_metric_group(self, metrics, keys, prefix=None, group_label=None):
        prefix_name = "" if prefix is None else f"{prefix}_"
        parts = []
        for key in keys:
            metric_key = f"{prefix_name}{key}"
            if metric_key in metrics:
                parts.append(f"{metric_key}: {metrics[metric_key]:.4f}")
        text = ", ".join(parts)
        if group_label is not None and text:
            return f"{group_label}: {text}"
        return text

    def format_ratio_pck_metrics(self, metrics, prefix=None, label=None):
        if not self.enable_ratio_pck:
            return ""
        keys = [ratio_to_key(ratio) for ratio in self.pck_threshold_ratios]
        return self._format_metric_group(metrics, keys, prefix=prefix, group_label=label)

    def format_pixel_pck_metrics(self, metrics, prefix=None, label=None):
        if not self.enable_pixel_pck:
            return ""
        keys = [pixel_to_key(pixel) for pixel in self.pck_threshold_pixels]
        return self._format_metric_group(metrics, keys, prefix=prefix, group_label=label)

    def format_pose_metric_lines(self, metrics, prefix=None, mpjpe_label=None, pixel_pck_label=None):
        lines = [self.format_mpjpe_metrics(metrics, prefix=prefix, label=mpjpe_label)]
        # Ratio-based PCK is temporarily disabled in logging/output.
        # Keep the code path for future reuse.
        # ratio_line = self.format_ratio_pck_metrics(metrics, prefix=prefix, label=ratio_pck_label)
        # if ratio_line:
        #     lines.append(ratio_line)
        pixel_line = self.format_pixel_pck_metrics(metrics, prefix=prefix, label=pixel_pck_label)
        if pixel_line:
            lines.append(pixel_line)
        return tuple(lines)

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
