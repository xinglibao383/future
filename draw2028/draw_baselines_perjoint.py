import matplotlib.pyplot as plt
import numpy as np
import os


def add_labels(x, y):
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi + 0.3,
            f"{yi:.0f}",
            ha='center',
            va='bottom',
            fontsize=10
        )


def plot_per_joint_group(save_path, labels, method_data):
    # plt.rcParams.update({
    #     "font.size": 12,
    #     "axes.labelsize": 12,
    #     "xtick.labelsize": 10,
    #     "ytick.labelsize": 10,
    #     "legend.fontsize": 10
    # })

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })

    methods = list(method_data.keys())
    num_methods = len(methods)

    group_gap = 1.1
    bar_width = 0.14
    x = np.arange(len(labels)) * group_gap
    offsets = (np.arange(num_methods) - (num_methods - 1) / 2) * bar_width

    plt.figure(figsize=(16, 5))

    all_values = []

    for i, method in enumerate(methods):
        values = method_data[method]
        x_pos = x + offsets[i]

        plt.bar(
            x_pos,
            values,
            width=bar_width,
            label=method
        )
        # add_labels(x_pos, values)
        all_values.extend(values)

    plt.xticks(x, labels, rotation=0, ha='center')
    plt.tick_params(axis='x', length=0)
    plt.xlabel("Joint")
    plt.ylabel("MPJPE")
    plt.ylim(0, 170)

    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.legend(frameon=False, ncol=4)

    # 收紧左右留白
    left_edge = x[0] + offsets[0] - bar_width / 2
    right_edge = x[-1] + offsets[-1] + bar_width / 2
    plt.xlim(left_edge - 0.08, right_edge + 0.08)

    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


def plot_all_per_joint_groups(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    labels = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar", "LBigToe",
        "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
    ]

    full_data = {
        "PIP": [
            87.5831, 53.3734, 55.8929, 59.6398, 91.9796,
            56.3072, 54.1846, 75.9508, 0.1033, 13.4973,
            54.1659, 77.1869, 14.9461, 53.9728, 70.8761,
            91.3041, 98.0057, 82.2365, 76.1647, 88.8291,
            91.3348, 76.3381, 84.2128, 84.4222, 81.4260
        ],
        "TIP": [
            90.9898, 58.2323, 60.6691, 64.6726, 97.4716,
            61.1391, 58.6436, 80.3936, 0.1250, 14.5020,
            58.1900, 82.5071, 15.9251, 58.3900, 78.2344,
            95.7595, 103.4124, 89.8814, 83.5982, 98.6341,
            100.3143, 85.3539, 89.1494, 89.8041, 85.2012
        ],
        "IMUPoser": [
            119.6239, 72.1608, 75.7613, 76.3657, 116.1634,
            76.4979, 70.8234, 98.2993, 0.0916, 17.8547,
            72.6780, 106.3654, 19.4749, 74.3524, 98.1636,
            126.1896, 139.0754, 125.1818, 112.7727, 127.8006,
            130.1094, 104.1137, 124.2156, 123.8907, 109.6657
        ],
        "DynaIP": [
            99.7378, 62.0194, 64.4993, 65.7845, 102.5599,
            65.0126, 60.4169, 81.3472, 0.1013, 14.6440,
            63.3735, 87.3093, 16.1928, 63.1966, 83.5251,
            100.9485, 112.8655, 93.1500, 86.6044, 101.5691,
            105.8163, 89.4580, 96.7746, 97.5194, 90.1578
        ],
        "ASIP": [
            88.2937, 57.6890, 60.2783, 63.0141, 96.1147,
            60.7454, 57.8683, 81.5145, 0.1237, 13.6544,
            57.9485, 82.4309, 15.5121, 58.1550, 75.9515,
            92.6118, 103.3944, 85.2642, 83.3824, 95.2122,
            98.6151, 83.5302, 88.0452, 89.6274, 84.1470
        ],
        "MobilePoser": [
            125.0026, 72.8326, 77.4007, 78.0088, 116.3247,
            79.0184, 74.4091, 101.9529, 0.1287, 19.4790,
            77.4240, 111.5459, 21.1892, 78.2185, 100.7290,
            135.8306, 142.8988, 129.4154, 116.2933, 132.7475,
            133.8030, 109.0107, 131.3712, 131.2569, 116.2548
        ],
        "AIPose (Ours)": [
            64.1483, 38.5916, 40.2098, 47.0918, 72.0981,
            40.3357, 42.3335, 60.4632, 0.2224, 11.4034,
            41.6569, 61.9996, 12.5581, 40.7128, 56.6253,
            71.3508, 76.5652, 66.2882, 60.0897, 71.1084,
            71.2205, 58.9327, 67.5625, 67.0125, 64.4659
        ]
    }

    group_indices = [
        (0, 13),   # 前13个
        (13, 25),  # 后12个
    ]

    for idx, (start, end) in enumerate(group_indices, start=1):
        current_labels = labels[start:end]
        method_data = {
            method: values[start:end]
            for method, values in full_data.items()
        }

        save_path = os.path.join(save_dir, f"per_joint_group_{idx}.png")
        plot_per_joint_group(
            save_path=save_path,
            labels=current_labels,
            method_data=method_data
        )


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/draw_baselines_perjoint.py
if __name__ == "__main__":
    # plot_all_per_joint_groups("/mnt/mydata/yh/liming/workspace/future/draw2028/imgs/per_joint")
    plot_all_per_joint_groups("/root/future/draw2028/imgs/per_joint")