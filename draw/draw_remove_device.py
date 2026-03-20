import matplotlib.pyplot as plt
import numpy as np


def add_labels(x, y):
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi + 0.3,                  # 稍微往上偏一点
            f"{yi:.2f}",               # 保留四位小数
            ha='center',
            va='bottom',
            fontsize=10
        )


def plot_remove_device(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12
    })

    labels = [
        "-",
        "0",
        "1",
        "2",
        "3",
        "4",
        "2,4"
    ]
    recon = [
        50.7079,
        52.2946,
        51.2945,
        50.5097,
        51.3100,
        50.6537,
        52.5957
    ]
    pred = [
        56.6010,
        58.6593,
        57.8350,
        56.7976,
        57.8461,
        56.2470,
        58.7247
    ]

    group_gap = 1.2
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
    plt.xlabel("Ablation Devices")
    plt.ylabel("MPJPE")
    plt.ylim(47, 61.9)
    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_remove_device("/mnt/mydata/yh/liming/workspace/future/draw/imgs/remove_device.png")