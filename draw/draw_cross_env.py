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


def plot_cross_scene(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    scenes = ["Bedroom", "Study", "Dining Room"]
    recon = [204.4093, 147.1609, 138.3596]
    pred  = [226.1702, 155.1837, 143.9825]

    group_gap = 1.3
    bar_width = 0.5
    x = np.arange(len(scenes)) * group_gap
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
    plt.xticks(x, scenes)
    plt.xlabel("Scene")
    plt.ylabel("MPJPE")
    plt.ylim(113, 243)
    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_cross_scene("/mnt/mydata/yh/liming/workspace/future/draw/imgs/cross_env.png")