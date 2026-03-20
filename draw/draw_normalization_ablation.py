import matplotlib.pyplot as plt
import numpy as np


def add_labels(x, y):
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi + 0.5,                  # 稍微往上偏一点
            f"{yi:.2f}",               # 保留四位小数
            ha='center',
            va='bottom',
            fontsize=10
        )


def plot_normalization_ablation(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12
    })

    labels = [
        "IMU✗ Pose✗",
        "IMU✗ Pose✓",
        "IMU✓ Pose✗",
        "IMU✓ Pose✓"
    ]
    recon = [60.7588, 79.7920, 58.5465, 50.6833]
    pred  = [74.1964, 86.5976, 70.5447, 56.4806]

    group_gap = 1.3
    bar_width = 0.5
    x = np.arange(len(labels)) * group_gap
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
    add_labels(x_recon, recon)
    add_labels(x_pred, pred)
    plt.xticks(x, labels)
    plt.xlabel("Normalization")
    plt.ylabel("MPJPE")
    plt.ylim(43, 93)
    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_normalization_ablation("/mnt/mydata/yh/liming/workspace/future/draw/imgs/normalization_ablation.png")