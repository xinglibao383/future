import os
import re
from typing import Dict, List, Optional, Union


class BaselineResult(dict):
    """
    {
        "method": "PIP",
        "best_epoch": 11,
        "current_mjae": 10.8673,
        "current_pe": 9.8989,
        "log_path": "...",
    }
    """
    pass


class AIPoseResult(dict):
    """
    {
        "method": "本文方法",
        "best_epoch": 11,
        "current_mjae": 10.8673,
        "current_pe": 9.8989,
        "future_mjae": 11.1022,
        "future_pe": 10.0311,
        "log_path": "...",
    }
    """
    pass


ParsedResult = Union[BaselineResult, AIPoseResult]


def normalize_method_name(raw_name: str) -> str:
    raw_name = raw_name.strip().lower()
    if raw_name == "aipose":
        return "本文方法"

    mapping = {
        "pip_like_recon": "PIP",
        "tip_like_recon": "TIP",
        "imuposer_like_recon": "IMUPoser",
        "dynaip_like_recon": "DynaIP",
        "asip_like_recon": "ASIP",
        "mobileposer_like_recon": "MobilePoser",
    }
    return mapping.get(raw_name, raw_name)


def read_text_file(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def parse_method_name(content: str) -> Optional[str]:
    match = re.search(r"\bbaseline\s*=\s*([A-Za-z0-9_]+)", content)
    if match:
        return normalize_method_name(match.group(1))

    if "baseline=aipose" in content.lower():
        return normalize_method_name("aipose")

    return None


def parse_best_epoch(content: str) -> Optional[int]:
    """
    必须匹配到类似：
    The best mean joint angle error occurred in epoch 11
    或
    The best mpjpe occurred in epoch 11
    """
    match = re.search(r"The best .*? occurred in epoch\s+(\d+)", content)
    if match:
        return int(match.group(1))
    return None


def parse_aipose_val_metrics_by_epoch(content: str) -> Dict[int, Dict[str, float]]:
    """
    解析本文方法在每个 epoch 的 val 指标。
    返回:
    {
        epoch: {
            "current_mjae": ...,
            "current_pe": ...,
            "future_mjae": ...,
            "future_pe": ...,
        }
    }
    """
    results: Dict[int, Dict[str, float]] = {}

    pattern = re.compile(
        r"Epoch:\s*(\d+),\s*"
        r"val loss1:\s*[0-9]*\.?[0-9]+,\s*"
        r"val loss2:\s*[0-9]*\.?[0-9]+,\s*"
        r"val loss3:\s*[0-9]*\.?[0-9]+,\s*"
        r"val current mean joint angle error:\s*([0-9]*\.?[0-9]+)\s*deg,\s*"
        r"val current positional error:\s*([0-9]*\.?[0-9]+)\s*cm,\s*"
        r"val future mean joint angle error:\s*([0-9]*\.?[0-9]+)\s*deg,\s*"
        r"val future positional error:\s*([0-9]*\.?[0-9]+)\s*cm"
    )

    for m in pattern.finditer(content):
        epoch = int(m.group(1))
        results[epoch] = {
            "current_mjae": float(m.group(2)),
            "current_pe": float(m.group(3)),
            "future_mjae": float(m.group(4)),
            "future_pe": float(m.group(5)),
        }

    return results


def parse_baseline_val_metrics_by_epoch(content: str) -> Dict[int, Dict[str, float]]:
    """
    解析 baseline 在每个 epoch 的 val 指标。
    返回:
    {
        epoch: {
            "current_mjae": ...,
            "current_pe": ...,
        }
    }
    """
    results: Dict[int, Dict[str, float]] = {}

    pattern = re.compile(
        r"Epoch:\s*(\d+),\s*"
        r"val loss:\s*[0-9]*\.?[0-9]+,\s*"
        r"val mean joint angle error:\s*([0-9]*\.?[0-9]+)\s*deg,\s*"
        r"val positional error:\s*([0-9]*\.?[0-9]+)\s*cm"
    )

    for m in pattern.finditer(content):
        epoch = int(m.group(1))
        results[epoch] = {
            "current_mjae": float(m.group(2)),
            "current_pe": float(m.group(3)),
        }

    return results


def parse_txt_file(txt_path: str) -> Optional[ParsedResult]:
    """
    严格解析单个日志文件。
    只有当：
    1) 能识别方法名
    2) 能匹配到 The best ... occurred in epoch X
    3) 能解析出 best_epoch 对应的 val 指标
    时才返回结果，否则返回 None
    """
    content = read_text_file(txt_path)
    if not content.strip():
        return None

    method_name = parse_method_name(content)
    if method_name is None:
        return None

    best_epoch = parse_best_epoch(content)
    if best_epoch is None:
        return None

    if method_name == "本文方法":
        val_metrics = parse_aipose_val_metrics_by_epoch(content)
        if best_epoch not in val_metrics:
            return None

        metrics = val_metrics[best_epoch]
        return AIPoseResult({
            "method": method_name,
            "best_epoch": best_epoch,
            "current_mjae": metrics["current_mjae"],
            "current_pe": metrics["current_pe"],
            "future_mjae": metrics["future_mjae"],
            "future_pe": metrics["future_pe"],
            # "log_path": txt_path,
        })

    val_metrics = parse_baseline_val_metrics_by_epoch(content)
    if best_epoch not in val_metrics:
        return None

    metrics = val_metrics[best_epoch]
    return BaselineResult({
        "method": method_name,
        "best_epoch": best_epoch,
        "current_mjae": metrics["current_mjae"],
        "current_pe": metrics["current_pe"],
        # "log_path": txt_path,
    })


def collect_results(root_dir: str) -> List[ParsedResult]:
    """
    递归遍历 root_dir 下所有 txt 文件。
    如果遇到同一种方法，保留后找到的那个。
    最终按固定顺序返回。
    """
    results_by_method: Dict[str, ParsedResult] = {}

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

            results_by_method[parsed["method"]] = parsed

    method_order = ["PIP", "TIP", "IMUPoser", "DynaIP", "ASIP", "MobilePoser", "本文方法"]

    results: List[ParsedResult] = []
    for method in method_order:
        if method in results_by_method:
            results.append(results_by_method[method])

    for method in sorted(results_by_method.keys()):
        if method not in method_order:
            results.append(results_by_method[method])

    return results


def sort_results_by_sum_desc(results: List[ParsedResult]) -> List[ParsedResult]:
    """
    对解析后的结果列表排序：
    排序规则：(current_mjae + current_pe) 求和 → 降序（大的在前）
    无论 BaselineResult / AIPoseResult 都支持
    """
    # 按和降序排列：reverse=True 表示大的在前
    sorted_results = sorted(
        results,
        key=lambda x: x["current_mjae"] + x["current_pe"],
        reverse=False
    )
    return sorted_results


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/draw2028/parse_dataset.py
if __name__ == "__main__":
    print("========================= DIP-IMU =========================")
    root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/dataset/DIP-IMU"
    results = sort_results_by_sum_desc(collect_results(root_dir))
    for item in results:
        print(item)

    print("========================= AMASS =========================")

    root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/dataset/AMASS"
    results = sort_results_by_sum_desc(collect_results(root_dir))
    for item in results:
        print(item)