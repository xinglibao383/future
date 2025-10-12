import json
from datetime import datetime
from collections import Counter

# 你的 JSON 文件路径
json_path = "/data/xinglibao/xrfv2/pose/2_kitchen_3.json"

# 读取 JSON 文件
with open(json_path, "r") as f:
    data = json.load(f)

# 用于统计每秒帧数
frame_per_second = Counter()

# 遍历每个 frame
for item in data:
    time_str = item["frame_time"]
    # 转换为 datetime 对象
    t = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    # 按秒取整（去掉毫秒）
    second_str = t.strftime("%Y-%m-%d %H:%M:%S")
    frame_per_second[second_str] += 1

# 输出统计结果（按时间排序）
for second, count in sorted(frame_per_second.items()):
    print(f"{second} : {count} 帧")