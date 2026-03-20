import matplotlib.pyplot as plt
import numpy as np


def plot_cross_subject(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    subjects = list(range(16))
    recon = [
        93.9199, 117.3705, 86.6879, 102.1780,
        126.7194, 104.7901, 100.7176, 108.6058,
        114.5809, 98.3641, 97.3505, 105.0606,
        85.6985, 95.7136, 103.5153, 115.2117
    ]
    pred = [
        100.6215, 123.4515, 91.1092, 108.2353,
        138.4886, 113.8852, 106.4436, 118.8354,
        122.3770, 106.3886, 104.0960, 112.9244,
        89.5146, 102.6839, 110.9676, 125.8084
    ]

    group_gap = 1.1      # 不同志愿者之间的间距（>1 就有空隙）
    bar_width = 0.5      # 每根柱子的宽度
    x = np.arange(len(subjects)) * group_gap
    x_recon = x - bar_width / 2
    x_pred  = x + bar_width / 2
    plt.figure(figsize=(8, 5))
    plt.bar(
        x_recon,
        recon,
        width=bar_width,
        label="Reconstruction"
    )
    plt.bar(
        x_pred,
        pred,
        width=bar_width,
        label="Prediction"
    )
    plt.xticks(x, subjects)
    plt.xlabel("Subject ID")
    plt.ylabel("MPJPE")
    plt.ylim(83, 143)
    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_cross_subject("/mnt/mydata/yh/liming/workspace/future/draw/cross_subject.png")