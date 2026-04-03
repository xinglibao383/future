import os
import re
from typing import List, Tuple, Optional, Dict


ResultTuple = Tuple[
    int,    # cross_idx
    float,  # current mpjpe
    float,  # current_pck@5px
    float,  # current_pck@10px
    float,  # current_pck@20px
    float,  # future mpjpe
    float,  # future_pck@5px
    float,  # future_pck@10px
    float,  # future_pck@20px
]


def parse_txt_file(txt_path: str) -> Optional[ResultTuple]:
    """
    解析单个 txt 文件。
    只有当：
    1) 第一行能解析出 cross_idx
    2) 文件中存在 'The best mpjpe occurred in epoch'
    3) 能解析出 best val mpjpe 和 best val pck
    时才返回结果，否则返回 None
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # 某些日志可能不是 utf-8，尝试降级读取
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    lines = content.splitlines()
    if not lines:
        return None

    first_line = lines[0]

    # 解析第一行的 cross_idx
    cross_idx_match = re.search(r"\bcross_idx\s*=\s*(\d+)\b", first_line)
    if not cross_idx_match:
        return None
    cross_idx = int(cross_idx_match.group(1))

    # 必须包含该标记
    if "The best mpjpe occurred in epoch" not in content:
        return None

    # 解析 best val mpjpe
    mpjpe_match = re.search(
        r"best val current mpjpe:\s*([0-9]*\.?[0-9]+)\s*,\s*best val future mpjpe:\s*([0-9]*\.?[0-9]+)",
        content
    )
    if not mpjpe_match:
        return None

    current_mpjpe = float(mpjpe_match.group(1))
    future_mpjpe = float(mpjpe_match.group(2))

    # 解析 best val pck
    pck_match = re.search(
        r"best val current pixel pck:\s*"
        r"current_pck@5px:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"current_pck@10px:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"current_pck@20px:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"best val future pixel pck:\s*"
        r"future_pck@5px:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"future_pck@10px:\s*([0-9]*\.?[0-9]+)\s*,\s*"
        r"future_pck@20px:\s*([0-9]*\.?[0-9]+)",
        content
    )
    if not pck_match:
        return None

    current_pck_5 = float(pck_match.group(1))
    current_pck_10 = float(pck_match.group(2))
    current_pck_20 = float(pck_match.group(3))
    future_pck_5 = float(pck_match.group(4))
    future_pck_10 = float(pck_match.group(5))
    future_pck_20 = float(pck_match.group(6))

    return (
        cross_idx,
        current_mpjpe,
        current_pck_5,
        current_pck_10,
        current_pck_20,
        future_mpjpe,
        future_pck_5,
        future_pck_10,
        future_pck_20,
    )


def collect_cross_person_results(root_dir: str) -> List[ResultTuple]:
    """
    递归遍历 root_dir 下所有 txt 文件，提取目标结果。
    如果遇到相同 cross_idx，保留后找到的那个文件。
    最终按 cross_idx 升序返回。
    """
    results_by_cross_idx: Dict[int, ResultTuple] = {}

    for current_root, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            if not filename.lower().endswith(".txt"):
                continue

            txt_path = os.path.join(current_root, filename)
            parsed = parse_txt_file(txt_path)
            if parsed is None:
                continue

            cross_idx = parsed[0]
            # 后找到的覆盖先找到的
            results_by_cross_idx[cross_idx] = parsed

    return [results_by_cross_idx[k] for k in sorted(results_by_cross_idx.keys())]


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/draw_cross_subject.py
if __name__ == "__main__":
    root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/cross_person"
    results = collect_cross_person_results(root_dir)
    print("cross_idx,current_mpjpe,current_pck@5px,current_pck@10px,current_pck@20px,future_mpjpe,future_pck@5px,future_pck@10px,future_pck@20px")
    for row in results:
        print(",".join(map(str, row)))




# import matplotlib.pyplot as plt
# import numpy as np


# def add_labels(x, y):
#     for xi, yi in zip(x, y):
#         plt.text(
#             xi,
#             yi + 0.5,                  # 稍微往上偏一点
#             f"{yi:.0f}",               # 保留四位小数
#             ha='center',
#             va='bottom',
#             fontsize=10
#         )


# def plot_cross_subject(save_path):
#     plt.rcParams.update({
#         "font.size": 12,
#         "axes.labelsize": 12,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "legend.fontsize": 10
#     })

#     subjects = list(range(16))
#     recon = [
#         93.9199, 117.3705, 86.6879, 102.1780,
#         126.7194, 104.7901, 100.7176, 108.6058,
#         114.5809, 98.3641, 97.3505, 105.0606,
#         85.6985, 95.7136, 103.5153, 115.2117
#     ]
#     pred = [
#         100.6215, 123.4515, 91.1092, 108.2353,
#         138.4886, 113.8852, 106.4436, 118.8354,
#         122.3770, 106.3886, 104.0960, 112.9244,
#         89.5146, 102.6839, 110.9676, 125.8084
#     ]

#     group_gap = 1.1      # 不同志愿者之间的间距（>1 就有空隙）
#     bar_width = 0.5      # 每根柱子的宽度
#     x = np.arange(len(subjects)) * group_gap
#     x_recon = x - bar_width / 2
#     x_pred  = x + bar_width / 2
#     plt.figure(figsize=(8, 5))
#     plt.bar(
#         x_recon,
#         recon,
#         width=bar_width,
#         label="Reconstruction"
#     )
#     plt.bar(
#         x_pred,
#         pred,
#         width=bar_width,
#         label="Prediction"
#     )
#     add_labels(x_recon, recon)
#     add_labels(x_pred, pred)
#     plt.xticks(x, subjects)
#     plt.xlabel("Subject ID")
#     plt.ylabel("MPJPE")
#     plt.ylim(83, 143)
#     plt.grid(True, axis='y', linestyle="--", alpha=0.4)
#     plt.legend(frameon=False)
#     plt.savefig(save_path, dpi=900, bbox_inches='tight')
#     plt.close()
#     print(f"已保存到: {save_path}")


# if __name__ == "__main__":
#     plot_cross_subject("/mnt/mydata/yh/liming/workspace/future/draw/imgs/cross_subject.png")