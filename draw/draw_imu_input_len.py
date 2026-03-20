import matplotlib.pyplot as plt


def plot_imu_input_len(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12
    })

    x = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
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

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(
        x, recon,
        marker='o',
        linewidth=1.5,
        label="Reconstruction"
    )
    line2, = plt.plot(
        x, pred,
        marker='o',
        linewidth=1.5,
        label="Prediction"
    )
    for xi, yi in zip(x, recon):
        plt.text(
            xi,
            yi + 0.8,
            f"{yi:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    for xi, yi in zip(x, pred):
        plt.text(
            xi,
            yi + 0.8,
            f"{yi:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    # ===== 坐标轴 =====
    plt.xlim(8, 157)
    plt.ylim(43, 88)

    plt.xlabel("IMU Input Length")
    plt.ylabel("MPJPE")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(
        handles=[line1, line2],
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False
    )
    plt.subplots_adjust(top=0.8)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_imu_input_len("/mnt/mydata/yh/liming/workspace/future/draw/imgs/imu_input_len.png")