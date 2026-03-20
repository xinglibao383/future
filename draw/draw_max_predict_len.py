import matplotlib.pyplot as plt


def plot_prediction_horizon(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12
    })

    x = [
        -7.5,   # [-15, 0)
        7.5,    # [0, 15)
        22.5,
        37.5,
        52.5,
        67.5,
        82.5,
        97.5,
        112.5,
        127.5,
        142.5
    ]

    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    x = [v / 15 for v in x]
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""

    y = [
        56.1941,
        60.9056,
        65.6229,
        77.7804,
        88.2202,
        94.8827,
        101.9103,
        108.3201,
        113.4193,
        119.1812,
        125.6723
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y,
        marker='o',
        linewidth=1.5,
        label="MPJPE"
    )
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi + 1,              # 稍微往上偏一点（关键）
            f"{yi:.2f}",         # 保留四位小数
            ha='center',
            va='bottom',
            fontsize=10
        )
    plt.xlim(-1.5, 10.5)
    plt.ylim(48, 132)
    plt.xlabel("Prediction Time (s)")
    plt.ylabel("MPJPE")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_prediction_horizon(save_path="/mnt/mydata/yh/liming/workspace/future/draw/imgs/max_predict_len.png")