import os
import json
import h5py
import numpy as np


def process_raw_imu_data(source_dir, target_dir):
    mapping = {'1': 'livingroom', '2': 'office', '3': 'kitchen'}
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        with h5py.File(os.path.join(source_dir, filename), 'r') as f:
            data = f['data'][()]
        name_parts = filename.replace('.h5', '').split('_')
        name_parts[1] = mapping[name_parts[1]]
        np.save(os.path.join(target_dir, '_'.join(name_parts) + '.npy'), data)


def process_raw_pose_data(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(source_dir, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)

            all_frames = []
            for frame in data:
                key = "pose_key_points:"
                if key in frame and frame[key]:
                    points = frame[key][0]
                    if len(points) != 25:
                        raise ValueError(f"{filename} 中关键点数量不足25个，实际为 {len(points)}")
                    all_frames.append(points)
                else:
                    all_frames.append([[0.0, 0.0, 0.0]] * 25)

            array = np.array(all_frames, dtype=np.float32)
            npy_filename = os.path.splitext(filename)[0] + ".npy"
            np.save(os.path.join(target_dir, npy_filename), array)
            print(f"Processed {filename}, shape: {array.shape}")


def process(source_dir, window_size, stride):
    for filename in os.listdir(source_dir):
        if filename.endswith(".npy"):
            split_npy_to_windows(os.path.join(source_dir, filename), os.path.join(source_dir, f"sample_{window_size}_{stride}"), window_size, stride)

def split_npy_to_windows(input_path, output_dir, window_size, stride):
    data = np.load(input_path)
    total_frames = data.shape[0]

    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    window_count = 0
    for start in range(0, total_frames - window_size + 1, stride):
        window = data[start:start + window_size]
        output_filename = f"{base_filename}_{window_count}.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, window)
        window_count += 1

    print(f"共生成 {window_count} 个窗口，保存到 {output_dir}")


if __name__ == "__main__":
    # process("/home/xinglibao/workspace/data/future/pose", 90, 15)
    # process_raw_pose_data("/home/xinglibao/data/kitchen", "/home/xinglibao/workspace/future/data/pose_raw")
    process_raw_imu_data("/home/luohonglin/workspace/imu/", "/home/xinglibao/workspace/future/data/imu_raw")