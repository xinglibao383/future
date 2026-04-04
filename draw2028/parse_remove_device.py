import os
import re
from typing import List, Tuple, Optional, Dict


ResultTuple = Tuple[
    Tuple[int, ...],  # exclude_device_idx
    float,  # current mpjpe
    float,  # current_pck@5px
    float,  # current_pck@10px
    float,  # current_pck@20px
    float,  # future mpjpe
    float,  # future_pck@5px
    float,  # future_pck@10px
    float,  # future_pck@20px
]


def parse_exclude_device_idx(first_line: str) -> Optional[Tuple[int, ...]]:
    """
    解析第一行中的 exclude_device_idx = (0, 3, 4)
    """
    match = re.search(r"exclude_device_idx\s*=\s*\((.*?)\)", first_line)
    if not match:
        return None

    inside = match.group(1)  # "0, 3, 4"
    try:
        nums = tuple(int(x.strip()) for x in inside.split(",") if x.strip() != "")
        return nums
    except:
        return None


def parse_txt_file(txt_path: str) -> Optional[ResultTuple]:
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

    # 解析 exclude_device_idx
    exclude_device_idx = parse_exclude_device_idx(first_line)
    if exclude_device_idx is None:
        return None

    # 必须包含 best 标志
    if "The best mpjpe occurred in epoch" not in content:
        return None

    # MPJPE
    mpjpe_match = re.search(
        r"best val current mpjpe:\s*([0-9]*\.?[0-9]+)\s*,\s*best val future mpjpe:\s*([0-9]*\.?[0-9]+)",
        content
    )
    if not mpjpe_match:
        return None

    current_mpjpe = float(mpjpe_match.group(1))
    future_mpjpe = float(mpjpe_match.group(2))

    # PCK
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
        exclude_device_idx,
        current_mpjpe,
        current_pck_5,
        current_pck_10,
        current_pck_20,
        future_mpjpe,
        future_pck_5,
        future_pck_10,
        future_pck_20,
    )


def collect_exclude_device_results(root_dir: str) -> List[ResultTuple]:
    """
    递归解析所有 txt
    相同 exclude_device_idx → 后覆盖前
    """
    results_dict: Dict[Tuple[int, ...], ResultTuple] = {}

    for current_root, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            if not filename.endswith(".txt"):
                continue

            txt_path = os.path.join(current_root, filename)
            parsed = parse_txt_file(txt_path)
            if parsed is None:
                continue

            key = parsed[0]
            results_dict[key] = parsed  # 覆盖策略

    # 按 tuple 排序（字典序）
    return [results_dict[k] for k in sorted(results_dict.keys(), key=lambda x: (len(x), x))]


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/parse_remove_device.py
if __name__ == "__main__":
    root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/exclude_device_200epoch"
    results = collect_exclude_device_results(root_dir)

    for r in results:
        print(r)