import re
import matplotlib.pyplot as plt


log_map = {
    "Ours": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260320135010/20260320135010.txt",
    "PIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260319225624/20260319225624.txt",
    "TIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260319225659/20260319225659.txt",
    "IMUPoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260321025336/20260321025336.txt",
    "DynaIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260319232806/20260319232806.txt",
    "ASIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260319235308/20260319235308.txt",
    "MobilePoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline/20260321023032/20260321023032.txt",
}


def parse_log_ours(log_path):
    """适用于 Ours：有 mpjpe1 / mpjpe2"""
    epochs = []
    mpjpes = []
    pattern = re.compile(
        r"Epoch:\s*(\d+).*val mpjpe1:\s*([\d.]+).*val mpjpe2:\s*([\d.]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                mpjpes.append(float(match.group(3)))  # 用 mpjpe2
    return epochs, mpjpes


def parse_log_default(log_path):
    epochs = []
    mpjpes = []

    pattern = re.compile(r"Epoch:\s*(\d+).*val mpjpe:\s*([\d.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                mpjpe = float(match.group(2))
                epochs.append(epoch)
                mpjpes.append(mpjpe)

    return epochs, mpjpes


def plot_baselines(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12
    })
    plt.figure(figsize=(8, 5))

    # ===== 存储线对象（用于legend）=====
    lines = []
    all_mpjpes = []

    for name, log_path in log_map.items():
        if name == "Ours":
            epochs, mpjpes = parse_log_ours(log_path)
        else:
            epochs, mpjpes = parse_log_default(log_path)

        if len(epochs) == 0:
            continue

        all_mpjpes.extend(mpjpes)

        # ===== 找最优（最小）MPJPE =====
        min_idx = mpjpes.index(min(mpjpes))
        min_val = mpjpes[min_idx]
        min_epoch = epochs[min_idx]

        # ===== 画曲线 + legend带最大值 =====
        line, = plt.plot(
            epochs,
            mpjpes,
            linewidth=1.5,
            label=f"{name}"
        )

        lines.append(line)

    # ===== 坐标轴 =====
    plt.xlim(-7, 207)

    # ===== y轴：动态范围 =====
    min_mpjpe = min(all_mpjpes)
    max_mpjpe = max(all_mpjpes)

    y_min = min_mpjpe - (max_mpjpe - min_mpjpe) * 0.05
    y_max = max_mpjpe + (max_mpjpe - min_mpjpe) * 0.05

    plt.ylim(y_min, y_max)

    plt.xlabel("Epoch")
    plt.ylabel("MPJPE")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(
        handles=lines,
        frameon=False
    )

    # ===== 保存 =====
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()

    print(f"已保存到: {save_path}")


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw/draw_baselines.py
if __name__ == "__main__":
    plot_baselines(save_path="/mnt/mydata/yh/liming/workspace/future/draw/imgs/Baseline Pose Reconstruction MPJPE Comparison.png")