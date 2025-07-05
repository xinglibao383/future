import os

def analyze_npy_filenames(folder_path):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    split_numbers = []

    for fname in filenames:
        name_parts = fname[:-4].split('_')  # 去掉 .npy 后缀并按下划线分割
        try:
            nums = [int(part) for part in name_parts]  # 或者 float(part) 如果是小数
            split_numbers.append(nums)
        except ValueError:
            print(f"跳过无法解析为数字的文件名: {fname}")

    if not split_numbers:
        print("没有有效的文件名可供分析。")
        return

    num_parts = len(split_numbers[0])
    for i in range(num_parts):
        ith_values = [parts[i] for parts in split_numbers if len(parts) > i]
        print(f"第 {i+1} 段：min={min(ith_values)}, max={max(ith_values)}")

# ✅ 使用示例
folder_path = "/home/xinglibao/workspace/future/datac/imu/imu_150_25"  # ← 修改为你的文件夹路径
analyze_npy_filenames(folder_path)