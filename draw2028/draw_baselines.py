import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
cn_font_label = FontProperties(fname=FONT_PATH, size=12)
cn_font_legend = FontProperties(fname=FONT_PATH, size=10)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


log_map = {
    "PIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175529/20260402175529.txt",
    "TIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175600/20260402175600.txt",
    "IMUPoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402181255/20260402181255.txt",
    "DynaIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175627/20260402175627.txt",
    "ASIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260403115422/20260403115422.txt",
    "MobilePoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402181241/20260402181241.txt",
    "本章方法": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402121744/20260402121744.txt",
}


# log_map = {
#     "PIP": "/root/future/outputs/experiment2028/baseline/20260402175529/20260402175529.txt",
#     "TIP": "/root/future/outputs/experiment2028/baseline/20260402175600/20260402175600.txt",
#     "IMUPoser": "/root/future/outputs/experiment2028/baseline/20260402181255/20260402181255.txt",
#     "DynaIP": "/root/future/outputs/experiment2028/baseline/20260402175627/20260402175627.txt",
#     "ASIP": "/root/future/outputs/experiment2028/baseline/20260403115422/20260403115422.txt",
#     "MobilePoser": "/root/future/outputs/experiment2028/baseline/20260402181241/20260402181241.txt",
#     "本章方法": "/root/future/outputs/experiment2028/baseline/20260402121744/20260402121744.txt",
# }


def parse_log_aipose(log_path):
    """
    适用于 AIPose:
    Epoch: x, val current mpjpe: ...
    """
    epochs = []
    mpjpes = []

    mpjpe_pattern = re.compile(
        r"Epoch:\s*(\d+).*val current mpjpe:\s*([\d.]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = mpjpe_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                mpjpes.append(float(match.group(2)))

    return epochs, mpjpes


def parse_log_default(log_path):
    """
    适用于其他基线:
    Epoch: x, val mpjpe: ...
    """
    epochs = []
    mpjpes = []

    mpjpe_pattern = re.compile(
        r"Epoch:\s*(\d+).*val mpjpe:\s*([\d.]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = mpjpe_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                mpjpes.append(float(match.group(2)))

    return epochs, mpjpes


def plot_baselines(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    lines = []
    all_mpjpes = []

    for name, log_path in log_map.items():
        if name == "本章方法":
            epochs, mpjpes = parse_log_aipose(log_path)
        else:
            epochs, mpjpes = parse_log_default(log_path)

        if len(epochs) == 0:
            print(f"[WARN] No valid epochs found in: {log_path}")
            continue

        all_mpjpes.extend(mpjpes)

        line, = ax.plot(
            epochs,
            mpjpes,
            linewidth=1.5,
            label=name
        )
        lines.append(line)

    if len(all_mpjpes) == 0:
        print("[ERROR] No MPJPE data parsed.")
        return

    ax.set_xlim(-8, 208)

    min_mpjpe = min(all_mpjpes)
    max_mpjpe = max(all_mpjpes)
    mpjpe_margin = (max_mpjpe - min_mpjpe) * 0.08 if max_mpjpe > min_mpjpe else 5
    ax.set_ylim(min_mpjpe - mpjpe_margin, max_mpjpe + mpjpe_margin)

    ax.set_xlabel("训练轮数（单位：轮）", fontproperties=cn_font_label)
    ax.set_ylabel("平均关节点位置误差（单位：像素）", fontproperties=cn_font_label)

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        handles=lines,
        frameon=False,
        loc="upper right",
        prop=cn_font_legend
    )

    plt.savefig(save_path, dpi=1500, bbox_inches="tight")
    plt.close()

    print(f"已保存到: {save_path}")


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/draw_baselines.py
if __name__ == "__main__":
    plot_baselines(save_path="/mnt/mydata/yh/liming/workspace/future/draw2028/imgs/aipose_baselines.png")
    # plot_baselines(save_path="/root/future/draw2028/imgs/aipose_baselines.png")