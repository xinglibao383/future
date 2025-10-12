import shutil
import os
import json
import h5py
import numpy as np


mapping = {'livingroom': '1', 'office': '2', 'kitchen': '3'}


def process(source_dir, target_dir, use_len, compute_len, predict_len, stride_len):
    imu_source_dir = os.path.join(source_dir, "imu")
    imu_target_dir = os.path.join(target_dir, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
    if os.path.exists(imu_target_dir):
        shutil.rmtree(imu_target_dir)
    os.makedirs(imu_target_dir)
    pose_source_dir = os.path.join(source_dir, "pose")
    pose_target_dir = os.path.join(target_dir, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
    if os.path.exists(pose_target_dir):
        shutil.rmtree(pose_target_dir)
    os.makedirs(pose_target_dir)
    
    for filename in os.listdir(pose_source_dir):
        name_parts = filename.replace('.json', '').split('_')
        name_parts[1] = mapping[name_parts[1]]
        imu_filepath = os.path.join(imu_source_dir, '_'.join(name_parts) + '.h5')
        if os.path.exists(imu_filepath):
            print(f"Processing: {imu_filepath}")
            with h5py.File(imu_filepath, 'r') as f:
                imu_data = f['data'][()]
            imu_data = np.transpose(imu_data, (0, 2, 1))
            imu_data = imu_data.reshape(-1, imu_data.shape[2])
            with open(os.path.join(pose_source_dir, filename), 'r') as f:
                pose_data = json.load(f)

            for frame_end_idx in range(use_len + compute_len + predict_len, len(pose_data), stride_len):
                pose_sub_data, imu_sub_data = [], []
                frame_start_idx = frame_end_idx - (use_len + compute_len + predict_len)
                for i in range(frame_start_idx, frame_end_idx):
                    points = pose_data[i]["pose_key_points:"]
                    if len(points) != 0:
                        pose_sub_data.append(points[0])
                    else:
                        pose_sub_data.append([[0.0, 0.0, 0.0]] * 25)
                imu_end_idx = int(imu_data.shape[1] / len(pose_data) * (frame_end_idx - (compute_len + predict_len)))
                imu_start_idx = imu_end_idx - int(use_len / 15 * 50)
                name_parts = filename.replace('.json', '').split('_')
                name_parts[1] = mapping[name_parts[1]]
                name_parts.append(str(frame_start_idx))
                np.save(os.path.join(imu_target_dir, '_'.join(name_parts) + '.npy'), imu_data[:, imu_start_idx:imu_end_idx])
                np.save(os.path.join(pose_target_dir, '_'.join(name_parts) + '.npy'), np.array(pose_sub_data, dtype=np.float32))
            

if __name__ == "__main__":
    process("/data/xinglibao/xrfv2", "/home/xinglibao/workspace/future/mydata", use_len=30, compute_len=15, predict_len=5, stride_len=15)