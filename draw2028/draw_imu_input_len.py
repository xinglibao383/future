import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
cn_font = FontProperties(fname=FONT_PATH, size=12)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_imu_input_len(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    x = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
    # 单位换算为秒，上面的是以视频帧为单位
    x = [v / 15 for v in x]

    recon = [
        78.1419, 63.0527, 56.0555, 50.7079,
        48.6763, 48.7303, 47.3098, 47.4721,
        48.6094, 47.9178
    ]
    pred = [
        81.7890, 69.1466, 63.0849, 56.6010,
        54.1706, 53.8068, 51.6719, 52.1078,
        53.8773, 52.9400
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    line1, = ax.plot(
        x, recon,
        marker='o',
        linewidth=1.5
    )
    line2, = ax.plot(
        x, pred,
        marker='o',
        linewidth=1.5
    )

    for xi, yi in zip(x, recon):
        ax.text(
            xi,
            yi + 0.8,
            f"{yi:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    for xi, yi in zip(x, pred):
        ax.text(
            xi,
            yi + 0.8,
            f"{yi:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_xlim(0.3, 10.7)
    ax.set_ylim(46, 86)
    ax.set_xlabel("IMU 序列输入长度（单位：秒）", fontproperties=cn_font)
    ax.set_ylabel("平均关节点位置误差（单位：像素）", fontproperties=cn_font)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend = ax.legend(
        [line1, line2],
        ["人体姿态重构", "人体姿态预测"],
        frameon=False,
        prop=cn_font
    )
    plt.savefig(save_path, dpi=1500, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    # plot_imu_input_len("/mnt/mydata/yh/liming/workspace/future/draw/imgs/imu_input_len.png")
    plot_imu_input_len("/root/future/draw2028/imgs/imu_input_len.png")