import shutil
import os
import json
import h5py
import numpy as np

# mapping = {'1': 'livingroom', '2': 'office', '3': 'kitchen'}
mapping = {'livingroom': '1', 'office': '2', 'kitchen': '3'}

def slice_with_padding(data, start, end):
    channels, time_len = data.shape
    slice_len = end - start + 1
    result = np.zeros((channels, slice_len), dtype=data.dtype)
    data_start = max(start, 0)
    data_end = min(end + 1, time_len)
    target_start = max(0, -start)
    target_end = target_start + (data_end - data_start)
    result[:, target_start:target_end] = data[:, data_start:data_end]
    return result

def process(source_dir, target_dir, imu_len):
    imu_source_dir = os.path.join(source_dir, "imu")
    imu_target_dir = os.path.join(target_dir, "imu")
    if os.path.exists(imu_target_dir):
        shutil.rmtree(imu_target_dir)
    os.makedirs(imu_target_dir)
    pose_source_dir = os.path.join(source_dir, "pose")
    pose_target_dir = os.path.join(target_dir, "pose")
    if os.path.exists(pose_target_dir):
        shutil.rmtree(pose_target_dir)
    os.makedirs(pose_target_dir)
    for filename in os.listdir(pose_source_dir):
        name_parts = filename.replace('.json', '').split('_')
        name_parts[1] = mapping[name_parts[1]]
        imu_filepath = os.path.join(imu_source_dir, '_'.join(name_parts) + '.h5')
        print(imu_filepath)
        if os.path.exists(imu_filepath):
            print(f"Processing: {imu_filepath}")
            with h5py.File(imu_filepath, 'r') as f:
                imu_data = f['data'][()]
            imu_data = np.transpose(imu_data, (0, 2, 1))
            imu_data = imu_data.reshape(-1, imu_data.shape[2])
            with open(os.path.join(pose_source_dir, filename), 'r') as f:
                pose_data = json.load(f)
            for frame in pose_data:
                if "frame_id" in frame and "pose_key_points:" in frame:
                    frame_id = frame["frame_id"]
                    name_parts = filename.replace('.json', '').split('_')
                    name_parts[1] = mapping[name_parts[1]]
                    name_parts.append(str(frame_id))
                    imu_end_index = int(imu_data.shape[1] / len(pose_data) * frame_id)
                    np.save(os.path.join(imu_target_dir, '_'.join(name_parts) + '.npy'), slice_with_padding(imu_data, imu_end_index-imu_len+1, imu_end_index))
                    np.save(os.path.join(pose_target_dir, '_'.join(name_parts) + '.npy'), np.array(frame["pose_key_points:"][0], dtype=np.float32))
        
if __name__ == "__main__":
    process("/data/xinglibao/xrfv2", "/home/xinglibao/workspace/future/mydata", 100)