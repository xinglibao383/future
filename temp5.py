import numpy as np

# 1. 指定你的 .npy 文件路径
file_path = "/home/xinglibao/workspace/future/data/imu_raw/0_kitchen_1.npy"  # 替换为你的文件名或路径

# 2. 读取 .npy 文件
data = np.load(file_path)

# 3. 打印数据的 shape
print("数据 shape:", data.shape)




# 1. 指定你的 .npy 文件路径
file_path = "/home/xinglibao/workspace/future/data/pose_raw/0_kitchen_1.npy"  # 替换为你的文件名或路径

# 2. 读取 .npy 文件
data = np.load(file_path)

# 3. 打印数据的 shape
print("数据 shape:", data.shape)