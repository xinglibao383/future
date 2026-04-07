import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
cn_font_label = FontProperties(fname=FONT_PATH, size=12)

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
    "AIPose (Ours)": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402121744/20260402121744.txt",
}


log_map = {
    "PIP": "/root/future/outputs/experiment2028/baseline/20260402175529/20260402175529.txt",
    "TIP": "/root/future/outputs/experiment2028/baseline/20260402175600/20260402175600.txt",
    "IMUPoser": "/root/future/outputs/experiment2028/baseline/20260402181255/20260402181255.txt",
    "DynaIP": "/root/future/outputs/experiment2028/baseline/20260402175627/20260402175627.txt",
    "ASIP": "/root/future/outputs/experiment2028/baseline/20260403115422/20260403115422.txt",
    "MobilePoser": "/root/future/outputs/experiment2028/baseline/20260402181241/20260402181241.txt",
    "AIPose (Ours)": "/root/future/outputs/experiment2028/baseline/20260402121744/20260402121744.txt",
}


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
        if "AIPose" in name or "Ours" in name:
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
        loc="upper right"
    )

    plt.savefig(save_path, dpi=1500, bbox_inches="tight")
    plt.close()

    print(f"已保存到: {save_path}")


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/draw_baselines.py
if __name__ == "__main__":
    # plot_baselines(save_path="/mnt/mydata/yh/liming/workspace/future/draw2028/imgs/aipose_baselines.png")
    plot_baselines(save_path="/root/future/draw2028/imgs/aipose_baselines.png")



# import re
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D


# log_map = {
#     "PIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175529/20260402175529.txt",
#     "TIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175600/20260402175600.txt",
#     "IMUPoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402181255/20260402181255.txt",
#     "DynaIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402175627/20260402175627.txt",
#     "ASIP": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260403115422/20260403115422.txt",
#     "MobilePoser": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402181241/20260402181241.txt",
#     "AIPose (Ours)": "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/baseline/20260402121744/20260402121744.txt",
# }


# def parse_log_aipose(log_path):
#     """
#     适用于 AIPose:
#     Epoch: x, val current mpjpe: ...
#     Epoch: x, val current pixel pck: current_pck@5px: ..., current_pck@10px: ..., current_pck@20px: ...
#     """
#     epochs = []
#     mpjpes = []
#     pck5s = []
#     pck10s = []
#     pck20s = []

#     mpjpe_pattern = re.compile(
#         r"Epoch:\s*(\d+).*val current mpjpe:\s*([\d.]+)"
#     )
#     pck_pattern = re.compile(
#         r"Epoch:\s*(\d+).*val current pixel pck:\s*"
#         r"current_pck@5px:\s*([\d.]+),\s*"
#         r"current_pck@10px:\s*([\d.]+),\s*"
#         r"current_pck@20px:\s*([\d.]+)"
#     )

#     mpjpe_dict = {}
#     pck_dict = {}

#     with open(log_path, "r", encoding="utf-8") as f:
#         for line in f:
#             m1 = mpjpe_pattern.search(line)
#             if m1:
#                 epoch = int(m1.group(1))
#                 mpjpe_dict[epoch] = float(m1.group(2))

#             m2 = pck_pattern.search(line)
#             if m2:
#                 epoch = int(m2.group(1))
#                 pck_dict[epoch] = (
#                     float(m2.group(2)) * 100.0,
#                     float(m2.group(3)) * 100.0,
#                     float(m2.group(4)) * 100.0,
#                 )

#     common_epochs = sorted(set(mpjpe_dict.keys()) & set(pck_dict.keys()))
#     for epoch in common_epochs:
#         epochs.append(epoch)
#         mpjpes.append(mpjpe_dict[epoch])
#         pck5s.append(pck_dict[epoch][0])
#         pck10s.append(pck_dict[epoch][1])
#         pck20s.append(pck_dict[epoch][2])

#     return epochs, mpjpes, pck5s, pck10s, pck20s


# def parse_log_default(log_path):
#     """
#     适用于其他基线:
#     Epoch: x, val mpjpe: ...
#     Epoch: x, val pixel pck: pck@5px: ..., pck@10px: ..., pck@20px: ...
#     """
#     epochs = []
#     mpjpes = []
#     pck5s = []
#     pck10s = []
#     pck20s = []

#     mpjpe_pattern = re.compile(
#         r"Epoch:\s*(\d+).*val mpjpe:\s*([\d.]+)"
#     )
#     pck_pattern = re.compile(
#         r"Epoch:\s*(\d+).*val pixel pck:\s*"
#         r"pck@5px:\s*([\d.]+),\s*"
#         r"pck@10px:\s*([\d.]+),\s*"
#         r"pck@20px:\s*([\d.]+)"
#     )

#     mpjpe_dict = {}
#     pck_dict = {}

#     with open(log_path, "r", encoding="utf-8") as f:
#         for line in f:
#             m1 = mpjpe_pattern.search(line)
#             if m1:
#                 epoch = int(m1.group(1))
#                 mpjpe_dict[epoch] = float(m1.group(2))

#             m2 = pck_pattern.search(line)
#             if m2:
#                 epoch = int(m2.group(1))
#                 pck_dict[epoch] = (
#                     float(m2.group(2)) * 100.0,
#                     float(m2.group(3)) * 100.0,
#                     float(m2.group(4)) * 100.0,
#                 )

#     common_epochs = sorted(set(mpjpe_dict.keys()) & set(pck_dict.keys()))
#     for epoch in common_epochs:
#         epochs.append(epoch)
#         mpjpes.append(mpjpe_dict[epoch])
#         pck5s.append(pck_dict[epoch][0])
#         pck10s.append(pck_dict[epoch][1])
#         pck20s.append(pck_dict[epoch][2])

#     return epochs, mpjpes, pck5s, pck10s, pck20s


# def plot_baselines_with_dual_axis(save_path):
#     plt.rcParams.update({
#         "font.size": 12,
#         "axes.titlesize": 12,
#         "axes.labelsize": 12,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "legend.fontsize": 10
#     })

#     fig, ax1 = plt.subplots(figsize=(11, 6))
#     ax2 = ax1.twinx()

#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#     all_mpjpes = []
#     all_pcks = []

#     method_handles = []

#     # 线型约定：同一种颜色表示同一个方法
#     # 实线：MPJPE
#     # 虚线：PCK@5
#     # 点划线：PCK@10
#     # 点线：PCK@20
#     pck_linestyles = {
#         "PCK@5px": "--",
#         "PCK@10px": "-.",
#         "PCK@20px": ":",
#     }

#     for idx, (name, log_path) in enumerate(log_map.items()):
#         color = color_cycle[idx % len(color_cycle)]

#         if "AIPose" in name or "Ours" in name:
#             epochs, mpjpes, pck5s, pck10s, pck20s = parse_log_aipose(log_path)
#         else:
#             epochs, mpjpes, pck5s, pck10s, pck20s = parse_log_default(log_path)

#         if len(epochs) == 0:
#             print(f"[WARN] No valid epochs found in: {log_path}")
#             continue

#         all_mpjpes.extend(mpjpes)
#         all_pcks.extend(pck5s)
#         all_pcks.extend(pck10s)
#         all_pcks.extend(pck20s)

#         # 左轴：MPJPE
#         line_mpjpe, = ax1.plot(
#             epochs,
#             mpjpes,
#             linestyle="-",
#             linewidth=2.0,
#             color=color,
#             alpha=0.95,
#         )

#         # 右轴：PCK
#         ax2.plot(
#             epochs,
#             pck5s,
#             linestyle=pck_linestyles["PCK@5px"],
#             linewidth=1.5,
#             color=color,
#             alpha=0.85,
#         )
#         ax2.plot(
#             epochs,
#             pck10s,
#             linestyle=pck_linestyles["PCK@10px"],
#             linewidth=1.5,
#             color=color,
#             alpha=0.85,
#         )
#         ax2.plot(
#             epochs,
#             pck20s,
#             linestyle=pck_linestyles["PCK@20px"],
#             linewidth=1.5,
#             color=color,
#             alpha=0.85,
#         )

#         method_handles.append(
#             Line2D([0], [0], color=color, lw=2.0, linestyle="-", label=name)
#         )

#     if len(all_mpjpes) == 0 or len(all_pcks) == 0:
#         print("[ERROR] No data parsed.")
#         return

#     # x 轴
#     ax1.set_xlim(-2, 205)
#     ax1.set_xlabel("Epoch")

#     # 左 y 轴：MPJPE
#     mpjpe_min, mpjpe_max = min(all_mpjpes), max(all_mpjpes)
#     mpjpe_margin = (mpjpe_max - mpjpe_min) * 0.08 if mpjpe_max > mpjpe_min else 5
#     ax1.set_ylim(mpjpe_min - mpjpe_margin, mpjpe_max + mpjpe_margin)
#     ax1.set_ylabel("MPJPE (px)")

#     # 右 y 轴：PCK
#     pck_min, pck_max = min(all_pcks), max(all_pcks)
#     pck_margin = (pck_max - pck_min) * 0.08 if pck_max > pck_min else 2
#     ax2.set_ylim(max(0, pck_min - pck_margin), min(100, pck_max + pck_margin))
#     ax2.set_ylabel("PCK (%)")

#     ax1.grid(True, linestyle="--", alpha=0.35)

#     # 图例1：方法
#     legend_methods = ax1.legend(
#         handles=method_handles,
#         loc="upper right",
#         frameon=False,
#         title="Methods"
#     )
#     ax1.add_artist(legend_methods)

#     # 图例2：指标线型
#     metric_handles = [
#         Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="MPJPE"),
#         Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PCK@5px"),
#         Line2D([0], [0], color="black", lw=2.0, linestyle="-.", label="PCK@10px"),
#         Line2D([0], [0], color="black", lw=2.0, linestyle=":", label="PCK@20px"),
#     ]
#     ax2.legend(
#         handles=metric_handles,
#         loc="lower right",
#         frameon=False,
#         title="Metrics"
#     )

#     plt.savefig(save_path, dpi=900, bbox_inches="tight")
#     plt.close()

#     print(f"已保存到: {save_path}")


# # /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/draw_baselines.py
# if __name__ == "__main__":
#     plot_baselines_with_dual_axis(save_path="/mnt/mydata/yh/liming/workspace/future/draw2028/imgs/aipose_baselines.png")