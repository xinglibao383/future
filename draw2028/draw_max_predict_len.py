from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
cn_font = FontProperties(fname=FONT_PATH, size=12)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_prediction_horizon(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
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
        47.8203,
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

    fig, ax = plt.subplots(figsize=(8, 5))

    line, = ax.plot(
        x,
        y,
        marker='o',
        linewidth=1.5,
        label="MPJPE"
    )

    for xi, yi in zip(x, y):
        ax.text(
            xi,
            yi + 1,
            f"{yi:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(43, 137)
    ax.set_xlabel("未来预测时间区间（单位：秒）", fontproperties=cn_font)
    ax.set_ylabel("平均关节点位置误差（单位：像素）", fontproperties=cn_font)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(save_path, dpi=1500, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    # plot_prediction_horizon(save_path="/mnt/mydata/yh/liming/workspace/future/draw/imgs/max_predict_len.png")
    plot_prediction_horizon(save_path="/root/future/draw2028/imgs/max_predict_len.png")