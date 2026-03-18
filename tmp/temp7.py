from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


SKELETON: Sequence[Tuple[int, int]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24),
)

DEFAULT_INPUT_PATH = Path("/mnt/mydata/yh/liming/workspace/future/mydata/pose/60_15_15_15")
DEFAULT_OUTPUT_DIRNAME = "pose_compare_vis"


@dataclass
class PoseProcessingResult:
    raw_pose: np.ndarray                 # (T, 25, 3)
    filled_pose: np.ndarray              # (T, 25, 3)
    normalized_pose: np.ndarray          # (T, 25, 3), tanh 后
    normalized_for_plot: np.ndarray      # (T, 25, 2), 用 atanh 反变换后的可视化坐标
    shoulder_width: np.ndarray           # (T,)
    raw_valid_mask: np.ndarray           # (T, 25)
    filled_valid_mask: np.ndarray        # (T, 25)


class PoseVisualizerError(RuntimeError):
    """Raised when the pose file format is unsupported."""


def canonicalize_pose_array(array: np.ndarray) -> np.ndarray:
    """
    将输入的 npy 统一整理成 (T, 25, C)，其中 C ∈ {2, 3}。
    兼容常见情况：
    - (T, 25, 3)
    - (25, 3)
    - (T, 75)
    - (75,)
    - 对于只有 (x, y) 的输入，也会补成 2 通道格式。
    """
    arr = np.asarray(array)

    if arr.ndim == 3 and arr.shape[1] == 25 and arr.shape[2] in (2, 3):
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 2:
        if arr.shape in ((25, 2), (25, 3)):
            return arr[None, ...].astype(np.float32, copy=False)
        if arr.shape[1] in (50, 75):
            channel_dim = arr.shape[1] // 25
            return arr.reshape(arr.shape[0], 25, channel_dim).astype(np.float32, copy=False)
        if arr.shape[0] in (50, 75) and arr.shape[1] == 1:
            channel_dim = arr.shape[0] // 25
            return arr.reshape(1, 25, channel_dim).astype(np.float32, copy=False)

    if arr.ndim == 1 and arr.size in (50, 75):
        channel_dim = arr.size // 25
        return arr.reshape(1, 25, channel_dim).astype(np.float32, copy=False)

    raise PoseVisualizerError(
        f"不支持的 pose 形状: {arr.shape}。期望形状类似 (T,25,3)、(25,3)、(T,75) 或 (75,)。"
    )


def ensure_confidence_channel(pose: np.ndarray) -> np.ndarray:
    """如果输入只有 (x, y)，则自动补一个全 1 的 confidence 通道。"""
    if pose.shape[-1] == 3:
        return pose.astype(np.float32, copy=False)

    confidence = np.ones((*pose.shape[:-1], 1), dtype=np.float32)
    return np.concatenate([pose.astype(np.float32, copy=False), confidence], axis=-1)


def fill_missing_keypoints(poses: np.ndarray, num_keypoints: int = 25) -> np.ndarray:
    """
    严格复现你给的逻辑：
    - 对于非最后一帧，如果关键点 confidence == 0，就向后找最近一个非零关键点补上；
    - 对于最后一帧，如果 confidence == 0，就向前找最近一个非零关键点补上。
    该函数会修改输入数组，因此外部应传入副本。
    """
    num_poses = poses.shape[0]

    for i in range(num_poses - 1):
        for j in range(num_keypoints):
            if poses[i, j, 2] == 0:
                for k in range(i + 1, num_poses):
                    if poses[k, j, 2] != 0:
                        poses[i, j] = poses[k, j]
                        break

    for j in range(num_keypoints):
        if poses[num_poses - 1, j, 2] == 0:
            for k in range(num_poses - 2, -1, -1):
                if poses[k, j, 2] != 0:
                    poses[num_poses - 1, j] = poses[k, j]
                    break

    return poses


def normalize_pose(
    keypoints: np.ndarray,
    center_idx: int = 8,
    left_shoulder_idx: int = 5,
    right_shoulder_idx: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    复现参考代码：
    1. 以第 8 号关键点为中心做平移；
    2. 用 5/2 两侧肩点距离做尺度归一化；
    3. 对归一化后的坐标施加 tanh。
    """
    centered = keypoints.copy()
    center = centered[:, center_idx:center_idx + 1, :2]
    centered[:, :, :2] -= center

    left_shoulder = centered[:, left_shoulder_idx, :2]
    right_shoulder = centered[:, right_shoulder_idx, :2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1).astype(np.float32)
    shoulder_width = np.clip(shoulder_width, a_min=1e-6, a_max=None)

    centered[:, :, :2] /= shoulder_width[:, None, None]
    centered[:, :, :2] = np.tanh(centered[:, :, :2])
    return centered, shoulder_width


def inverse_tanh_for_plot(normalized_xy: np.ndarray, clamp_value: float = 0.9999) -> np.ndarray:
    """
    复现你给的 plot_poses 可视化逻辑：
    先 clamp 到 (-1, 1)，再做 atanh，便于把 tanh 压缩前的归一化骨架形态画出来。
    """
    clamped = np.clip(normalized_xy, -clamp_value, clamp_value)
    return np.arctanh(clamped)


def load_and_process_pose(file_path: Path) -> PoseProcessingResult:
    pose = np.load(file_path)
    pose = canonicalize_pose_array(pose)
    pose = ensure_confidence_channel(pose)

    raw_pose = pose.copy()
    filled_pose = fill_missing_keypoints(pose.copy())
    normalized_pose, shoulder_width = normalize_pose(filled_pose.copy())
    normalized_for_plot = inverse_tanh_for_plot(normalized_pose[:, :, :2])
    raw_valid_mask = raw_pose[:, :, 2] > 0
    filled_valid_mask = filled_pose[:, :, 2] > 0

    return PoseProcessingResult(
        raw_pose=raw_pose,
        filled_pose=filled_pose,
        normalized_pose=normalized_pose,
        normalized_for_plot=normalized_for_plot,
        shoulder_width=shoulder_width,
        raw_valid_mask=raw_valid_mask,
        filled_valid_mask=filled_valid_mask,
    )


def robust_median(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return float(fallback)
    return float(np.median(finite))


def filter_finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points 期望形状为 (N,2)，但拿到了 {points.shape}")
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def compute_bounds(
    points: np.ndarray,
    padding_ratio: float,
    min_width: float,
    min_height: float,
    symmetric_center: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    计算统一的绘图边界。
    - 若 symmetric_center 为 None，则根据数据外接框做带 padding 的矩形；
    - 否则以给定中心做对称边界（处理后的归一化姿态更适合这样做）。
    """
    pts = filter_finite_points(points)
    if pts.size == 0:
        half_w = max(min_width / 2.0, 1.0)
        half_h = max(min_height / 2.0, 1.0)
        cx, cy = symmetric_center if symmetric_center is not None else (0.0, 0.0)
        return (cx - half_w, cx + half_w), (cy - half_h, cy + half_h)

    if symmetric_center is None:
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        width = max(float(xmax - xmin), min_width)
        height = max(float(ymax - ymin), min_height)
        cx = float((xmin + xmax) / 2.0)
        cy = float((ymin + ymax) / 2.0)
        half_w = width * (1.0 + padding_ratio) / 2.0
        half_h = height * (1.0 + padding_ratio) / 2.0
        return (cx - half_w, cx + half_w), (cy - half_h, cy + half_h)

    cx, cy = symmetric_center
    max_abs_x = max(float(np.max(np.abs(pts[:, 0] - cx))), min_width / 2.0)
    max_abs_y = max(float(np.max(np.abs(pts[:, 1] - cy))), min_height / 2.0)
    half_w = max_abs_x * (1.0 + padding_ratio)
    half_h = max_abs_y * (1.0 + padding_ratio)
    return (cx - half_w, cx + half_w), (cy - half_h, cy + half_h)


def estimate_raw_plot_box(raw_pose: np.ndarray, raw_valid_mask: np.ndarray, shoulder_width: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    valid_points = raw_pose[:, :, :2][raw_valid_mask]
    median_shoulder = robust_median(shoulder_width, fallback=50.0)

    # 这里的最小框大小按“约 4 个肩宽 × 6 个肩宽”估计，既不会太挤，也不会过空。
    min_width = 4.0 * median_shoulder
    min_height = 6.0 * median_shoulder
    return compute_bounds(
        points=valid_points.reshape(-1, 2) if valid_points.size > 0 else raw_pose[:, :, :2].reshape(-1, 2),
        padding_ratio=0.10,
        min_width=min_width,
        min_height=min_height,
        symmetric_center=None,
    )


def estimate_processed_plot_box(
    processed_pose_for_plot: np.ndarray,
    processed_valid_mask: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    valid_points = processed_pose_for_plot[processed_valid_mask]
    points = valid_points.reshape(-1, 2) if valid_points.size > 0 else processed_pose_for_plot.reshape(-1, 2)

    # 处理后已经以 8 号点为原点、以肩宽归一化，因此这里采用对称边界：
    # 至少保留 4(宽) × 6(高) 的标准化观察框，并再给 10% 边距。
    return compute_bounds(
        points=points,
        padding_ratio=0.10,
        min_width=4.0,
        min_height=6.0,
        symmetric_center=(0.0, 0.0),
    )


def draw_pose(
    ax: plt.Axes,
    pose_xy: np.ndarray,
    valid_mask: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    annotate_indices: bool = False,
) -> None:
    valid_mask = valid_mask.astype(bool)
    pose_xy = np.asarray(pose_xy, dtype=np.float32)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    # 保留坐标框与网格，满足“像坐标系一样有个框”的需求。
    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        fill=False,
        linewidth=1.5,
        edgecolor="black",
    )
    ax.add_patch(rect)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    for start_idx, end_idx in SKELETON:
        if valid_mask[start_idx] and valid_mask[end_idx]:
            x_pair = [pose_xy[start_idx, 0], pose_xy[end_idx, 0]]
            y_pair = [pose_xy[start_idx, 1], pose_xy[end_idx, 1]]
            if np.isfinite(x_pair).all() and np.isfinite(y_pair).all():
                ax.plot(x_pair, y_pair, linewidth=2)

    valid_points = pose_xy[valid_mask]
    if valid_points.size > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], s=28)

    if annotate_indices:
        for joint_idx, (x_coord, y_coord) in enumerate(pose_xy):
            if valid_mask[joint_idx] and np.isfinite(x_coord) and np.isfinite(y_coord):
                ax.text(x_coord + 0.02, y_coord + 0.02, str(joint_idx), fontsize=7)

    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_pose_comparison_figure(
    raw_pose_xy: np.ndarray,
    raw_valid_mask: np.ndarray,
    processed_pose_xy: np.ndarray,
    processed_valid_mask: np.ndarray,
    raw_box: Tuple[Tuple[float, float], Tuple[float, float]],
    processed_box: Tuple[Tuple[float, float], Tuple[float, float]],
    frame_idx: int,
    save_path: Path,
    file_stem: str,
    annotate_indices: bool,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))

    draw_pose(
        ax=axes[0],
        pose_xy=raw_pose_xy,
        valid_mask=raw_valid_mask,
        xlim=raw_box[0],
        ylim=raw_box[1],
        title="Before: raw pose",
        annotate_indices=annotate_indices,
    )
    draw_pose(
        ax=axes[1],
        pose_xy=processed_pose_xy,
        valid_mask=processed_valid_mask,
        xlim=processed_box[0],
        ylim=processed_box[1],
        title="After: normalized pose (atanh for display)",
        annotate_indices=annotate_indices,
    )

    fig.suptitle(f"{file_stem} | pose #{frame_idx:05d}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_pose_file(
    file_path: Path,
    output_root: Path,
    annotate_indices: bool = False,
    dpi: int = 220,
) -> dict:
    result = load_and_process_pose(file_path)
    file_output_dir = output_root / file_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    raw_box = estimate_raw_plot_box(
        raw_pose=result.raw_pose,
        raw_valid_mask=result.raw_valid_mask,
        shoulder_width=result.shoulder_width,
    )
    processed_box = estimate_processed_plot_box(
        processed_pose_for_plot=result.normalized_for_plot,
        processed_valid_mask=result.filled_valid_mask,
    )

    for frame_idx in range(result.raw_pose.shape[0]):
        save_path = file_output_dir / f"pose_{frame_idx:05d}.png"
        save_pose_comparison_figure(
            raw_pose_xy=result.raw_pose[frame_idx, :, :2],
            raw_valid_mask=result.raw_valid_mask[frame_idx],
            processed_pose_xy=result.normalized_for_plot[frame_idx],
            processed_valid_mask=result.filled_valid_mask[frame_idx],
            raw_box=raw_box,
            processed_box=processed_box,
            frame_idx=frame_idx,
            save_path=save_path,
            file_stem=file_path.stem,
            annotate_indices=annotate_indices,
            dpi=dpi,
        )

    summary = {
        "file": str(file_path),
        "num_frames": int(result.raw_pose.shape[0]),
        "output_dir": str(file_output_dir),
        "raw_box": {
            "xlim": [float(raw_box[0][0]), float(raw_box[0][1])],
            "ylim": [float(raw_box[1][0]), float(raw_box[1][1])],
        },
        "processed_box": {
            "xlim": [float(processed_box[0][0]), float(processed_box[0][1])],
            "ylim": [float(processed_box[1][0]), float(processed_box[1][1])],
        },
        "median_shoulder_width": robust_median(result.shoulder_width),
    }

    summary_path = file_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def iter_pose_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise PoseVisualizerError(f"输入文件不是 .npy: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise PoseVisualizerError(f"输入路径不存在或不是目录: {input_path}")

    for file_path in sorted(input_path.glob("*.npy")):
        yield file_path


def resolve_output_root(input_path: Path, explicit_output_root: Optional[Path]) -> Path:
    if explicit_output_root is not None:
        return explicit_output_root

    if input_path.is_file():
        return input_path.parent / DEFAULT_OUTPUT_DIRNAME
    return input_path / DEFAULT_OUTPUT_DIRNAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 pose npy，复现参考处理逻辑，并为每一帧保存处理前/处理后的对比图。"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="输入的 .npy 文件或包含多个 .npy 的目录。默认就是你给出的 pose 目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "输出根目录。默认保存在输入目录下的 pose_compare_vis/ 中；"
            "若输入是单个文件，则默认保存到该文件同级目录的 pose_compare_vis/。"
        ),
    )
    parser.add_argument(
        "--annotate-indices",
        action="store_true",
        help="是否在图上标出每个关键点编号。默认关闭，避免画面过挤。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图像的 DPI，默认 220。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path
    output_root = resolve_output_root(input_path, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    pose_files = list(iter_pose_files(input_path))
    if not pose_files:
        raise PoseVisualizerError(f"在 {input_path} 下没有找到任何 .npy 文件。")

    for file_path in pose_files:
        summary = process_pose_file(
            file_path=file_path,
            output_root=output_root,
            annotate_indices=args.annotate_indices,
            dpi=args.dpi,
        )
        summaries.append(summary)
        print(f"[OK] {file_path.name} -> {summary['output_dir']} ({summary['num_frames']} frames)")

    aggregate_summary_path = output_root / "all_files_summary.json"
    aggregate_summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] 对比图已保存到: {output_root}")


if __name__ == "__main__":
    main()
