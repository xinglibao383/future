import os
import re
from typing import List, Tuple, Optional, Dict


ResultTuple = Tuple[
    str,    # imu_generator
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
    1) 第一行能解析出 imu_generator
    2) 文件中存在 'The best mpjpe occurred in epoch'
    3) 能解析出 best val mpjpe 和 best val pck
    时才返回结果，否则返回 None
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    lines = content.splitlines()
    if not lines:
        return None

    first_line = lines[0]

    # 解析第一行的 imu_generator
    # 例如: [2026-04-03 17:04:06] 备注: 选择backbone, imu_generator=transformer
    imu_generator_match = re.search(r"\bimu_generator\s*=\s*([A-Za-z0-9_\-\.]+)\b", first_line)
    if not imu_generator_match:
        return None
    imu_generator = imu_generator_match.group(1)

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
        imu_generator,
        current_mpjpe,
        current_pck_5,
        current_pck_10,
        current_pck_20,
        future_mpjpe,
        future_pck_5,
        future_pck_10,
        future_pck_20,
    )


def collect_select_backbone_results(root_dir: str) -> List[ResultTuple]:
    """
    递归遍历 root_dir 下所有 txt 文件，提取目标结果。
    如果遇到相同 imu_generator，保留后找到的那个文件。
    最终按 imu_generator 字典序返回。
    """
    results_by_backbone: Dict[str, ResultTuple] = {}

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

            imu_generator = parsed[0]
            # 后找到的覆盖先找到的
            results_by_backbone[imu_generator] = parsed

    return [results_by_backbone[k] for k in sorted(results_by_backbone.keys())]


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/parse_backbone.py
if __name__ == "__main__":
    root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/select_backbone"
    results = collect_select_backbone_results(root_dir)

    for item in results:
        print(item)