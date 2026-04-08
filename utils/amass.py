import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from smplx import SMPL
except ImportError:
    SMPL = None


# 17个虚拟IMU安装点，对应 SMPL 24 body joints 的索引
# 这是一套工程上较容易使用的选择：
# pelvis, neck, head, shoulders, elbows, wrists, hips, knees, ankles, feet
SENSOR_JOINT_IDS = [
    0,   # pelvis
    3,   # spine2 / upper torso
    6,   # neck
    9,   # head
    13,  # left_collar / left shoulder region
    14,  # right_collar / right shoulder region
    16,  # left_shoulder
    17,  # right_shoulder
    18,  # left_elbow
    19,  # right_elbow
    20,  # left_wrist
    21,  # right_wrist
    1,   # left_hip
    2,   # right_hip
    4,   # left_knee
    5,   # right_knee
    7,   # left_ankle
]
# 如果你想改成更接近 DIP 17-sensor 的定义，可以只改这里


def axis_angle_to_matrix(axis_angle: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    axis-angle -> rotation matrix
    axis_angle: (..., 3)
    return: (..., 3, 3)
    """
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.clip(theta, eps, None)

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    zeros = np.zeros_like(x)

    k = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*axis.shape[:-1], 3, 3)

    eye = np.broadcast_to(np.eye(3, dtype=axis_angle.dtype), (*axis.shape[:-1], 3, 3))

    theta_expand = theta[..., None]   # (..., 1, 1)
    sin_theta = np.sin(theta_expand)
    cos_theta = np.cos(theta_expand)

    rot = eye + sin_theta * k + (1.0 - cos_theta) * (k @ k)

    small = (theta < 1e-6)[..., None]  # (..., 1, 1)
    rot = np.where(small, eye, rot)
    return rot


def resample_by_fps(arr: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """
    最近邻重采样
    arr: (T, ...)
    """
    if abs(src_fps - tgt_fps) < 1e-8:
        return arr

    old_len = arr.shape[0]
    new_len = int(round(old_len * float(tgt_fps) / float(src_fps)))
    new_len = max(new_len, 1)

    indices = np.round(np.arange(new_len) * float(src_fps) / float(tgt_fps)).astype(np.int64)
    indices = np.clip(indices, 0, old_len - 1)
    return arr[indices]


class SMPLForward:
    def __init__(self, model_dir: str, device: str = "cpu"):
        if SMPL is None:
            raise ImportError("smplx is not installed. Please install smplx first.")

        self.device = torch.device(device)

        self.models = {
            "male": SMPL(model_path=model_dir, gender="male", batch_size=1).to(self.device),
            "female": SMPL(model_path=model_dir, gender="female", batch_size=1).to(self.device),
            "neutral": SMPL(model_path=model_dir, gender="neutral", batch_size=1).to(self.device),
        }

    def get_model(self, gender: str):
        gender = str(gender).lower()
        if gender.startswith("m"):
            return self.models["male"]
        elif gender.startswith("f"):
            return self.models["female"]
        return self.models["neutral"]

    @torch.no_grad()
    def pose_to_joints(self, pose_axis_angle: np.ndarray, trans: np.ndarray, betas: np.ndarray, gender: str):
        """
        pose_axis_angle: (T, 24, 3)
        trans: (T, 3)
        betas: (10,) or (16,)
        return:
            joints: (T, J, 3)
        """
        model = self.get_model(gender)

        T = pose_axis_angle.shape[0]

        pose_tensor = torch.tensor(pose_axis_angle, dtype=torch.float32, device=self.device)
        trans_tensor = torch.tensor(trans, dtype=torch.float32, device=self.device)

        if betas.shape[0] >= 10:
            betas = betas[:10]
        betas_tensor = torch.tensor(betas, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(T, 1)

        global_orient = pose_tensor[:, 0, :]               # (T, 3)
        body_pose = pose_tensor[:, 1:, :].reshape(T, 23 * 3)

        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_tensor,
            transl=trans_tensor,
            return_verts=False,
        )

        joints = output.joints[:, :24, :].detach().cpu().numpy()   # 只取前24个body joints
        return joints


def synthesize_imu_from_amass(body_pose_24: np.ndarray, joints_24: np.ndarray, fps: int):
    """
    body_pose_24: (T, 24, 3)
    joints_24:    (T, 24, 3), 单位米
    return:
        imu_ori: (T, 17, 3, 3)
        imu_acc: (T, 17, 3)
        imu:     (T, 17, 12)
    """
    T = body_pose_24.shape[0]
    dt = 1.0 / fps

    # 用对应joint的局部旋转作为虚拟IMU方向
    joint_rot_all = axis_angle_to_matrix(body_pose_24)  # (T, 24, 3, 3)
    imu_ori = joint_rot_all[:, SENSOR_JOINT_IDS, :, :]  # (T, 17, 3, 3)

    # 用对应joint的世界坐标位置合成加速度
    sensor_pos = joints_24[:, SENSOR_JOINT_IDS, :]      # (T, 17, 3)

    imu_acc_world = np.zeros_like(sensor_pos, dtype=np.float32)
    if T >= 3:
        imu_acc_world[1:-1] = (sensor_pos[2:] - 2.0 * sensor_pos[1:-1] + sensor_pos[:-2]) / (dt * dt)
        imu_acc_world[0] = imu_acc_world[1]
        imu_acc_world[-1] = imu_acc_world[-2]
    elif T == 2:
        imu_acc_world[:] = 0.0
    else:
        imu_acc_world[:] = 0.0

    # 加上重力，再转到局部坐标系
    gravity = np.array([0.0, 0.0, 9.81], dtype=np.float32).reshape(1, 1, 3)
    imu_acc_with_g = imu_acc_world + gravity

    imu_acc = np.einsum("tsij,tsj->tsi", np.transpose(imu_ori, (0, 1, 3, 2)), imu_acc_with_g)
    imu_acc = imu_acc.astype(np.float32)

    imu_ori_flat = imu_ori.reshape(T, len(SENSOR_JOINT_IDS), 9).astype(np.float32)
    imu = np.concatenate([imu_ori_flat, imu_acc], axis=-1)  # (T, 17, 12)

    return imu_ori.astype(np.float32), imu_acc, imu.astype(np.float32)


def process(source_dir, target_dir, smpl_model_dir,
            use_len, compute_len, predict_len, stride_len,
            fps=60, subsets=None, device="cpu"):
    """
    将 AMASS 原始 .npz 数据转成虚拟IMU + pose，并切窗保存为:
        target_dir/
            imu/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy
            pose/{use_len}_{compute_len}_{predict_len}_{stride_len}/*.npy

    参数单位均为秒。
    最终输出与当前 DIP_IMU 数据格式兼容：
        imu sample:  (17*12, use_len_frames+predict_len_frames)
        pose sample: (compute_len_frames+predict_len_frames, 24, 3)
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"AMASS目录不存在: {source_dir}")
    if not os.path.exists(smpl_model_dir):
        raise FileNotFoundError(f"SMPL模型目录不存在: {smpl_model_dir}")

    imu_target_dir = os.path.join(target_dir, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
    pose_target_dir = os.path.join(target_dir, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")

    if os.path.exists(imu_target_dir):
        shutil.rmtree(imu_target_dir)
    if os.path.exists(pose_target_dir):
        shutil.rmtree(pose_target_dir)

    os.makedirs(imu_target_dir, exist_ok=True)
    os.makedirs(pose_target_dir, exist_ok=True)

    use_len_frames = int(use_len * fps)
    compute_len_frames = int(compute_len * fps)
    predict_len_frames = int(predict_len * fps)
    stride_len_frames = int(stride_len * fps)

    imu_total_len = use_len_frames + predict_len_frames
    pose_total_len = compute_len_frames + predict_len_frames

    smpl_forward = SMPLForward(smpl_model_dir, device=device)

    total_seq_count = 0
    total_count = 0

    if subsets is not None:
        subsets = set(subsets)

    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            if not fname.endswith(".npz"):
                continue

            filepath = os.path.join(root, fname)

            if subsets is not None:
                rel_path = os.path.relpath(filepath, source_dir)
                top_level = rel_path.split(os.sep)[0]
                if top_level not in subsets:
                    continue

            print(f"Processing: {filepath}")
            total_seq_count += 1

            data = np.load(filepath)

            required_keys = {"poses", "trans", "betas", "gender", "mocap_framerate"}
            if not required_keys.issubset(set(data.files)):
                print(f"Skip (missing keys): {filepath}")
                continue

            poses = np.asarray(data["poses"], dtype=np.float32)              # (T, 156)
            trans = np.asarray(data["trans"], dtype=np.float32)              # (T, 3)
            betas = np.asarray(data["betas"], dtype=np.float32)              # (16,)
            gender = str(np.asarray(data["gender"]))
            mocap_framerate = float(np.asarray(data["mocap_framerate"]))

            if poses.shape[1] < 72:
                print(f"Skip (poses dim < 72): {filepath}, shape={poses.shape}")
                continue

            # 取前24个body joints，对齐当前工程
            body_pose_24 = poses[:, :72].reshape(-1, 24, 3)   # (T, 24, 3)

            # 重采样
            body_pose_24 = resample_by_fps(body_pose_24, mocap_framerate, fps)
            trans = resample_by_fps(trans, mocap_framerate, fps)

            # 用SMPL前向得到24 body joints三维位置
            joints_24 = smpl_forward.pose_to_joints(body_pose_24, trans, betas, gender)  # (T, 24, 3)

            # 合成虚拟IMU
            _, _, imu = synthesize_imu_from_amass(body_pose_24, joints_24, fps)  # (T, 17, 12)

            seq_len = min(body_pose_24.shape[0], imu.shape[0])
            body_pose_24 = body_pose_24[:seq_len]
            imu = imu[:seq_len]

            max_start = seq_len - max(imu_total_len, pose_total_len)
            if max_start < 0:
                continue

            rel_path = os.path.relpath(filepath, source_dir)
            rel_path_no_ext = rel_path.replace(".npz", "").replace(os.sep, "__")

            for start_idx in range(0, max_start + 1, stride_len_frames):
                imu_start_idx = start_idx
                imu_end_idx = imu_start_idx + imu_total_len

                pose_start_idx = start_idx
                pose_end_idx = pose_start_idx + pose_total_len

                imu_sub_data = imu[imu_start_idx:imu_end_idx]            # (use+pred, 17, 12)
                pose_sub_data = body_pose_24[pose_start_idx:pose_end_idx]  # (compute+pred, 24, 3)

                # 转成与你当前代码一致的输入格式: (17*12, T)
                imu_sub_data = np.transpose(imu_sub_data, (1, 2, 0)).reshape(-1, imu_sub_data.shape[0])

                save_name = f"{rel_path_no_ext}_{start_idx}.npy"
                np.save(os.path.join(imu_target_dir, save_name), imu_sub_data.astype(np.float32))
                np.save(os.path.join(pose_target_dir, save_name), pose_sub_data.astype(np.float32))

                total_count += 1

    print("\n处理完成")
    print(f"原始序列数: {total_seq_count}")
    print(f"总样本数: {total_count}")
    print(f"IMU 保存目录: {imu_target_dir}")
    print(f"Pose 保存目录: {pose_target_dir}")


class AMASS_SYNTH_IMU(Dataset):
    def __init__(self, root_path, use_len, compute_len, predict_len, stride_len,
                 fps=60, source_dir="/mnt/mydata/yh/liming/data/AMASS",
                 smpl_model_dir="/mnt/mydata/yh/liming/workspace/future/SMPL",
                 subsets=None, device="cuda:1"):
        """
        use_len / compute_len / predict_len / stride_len 单位均为秒
        """
        super().__init__()
        root_path = "/mnt/mydata/yh/liming/workspace/future/mydata/AMASS_SYNTH_IMU_split"

        self.imu_root_path = os.path.join(root_path, "imu", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")
        self.pose_root_path = os.path.join(root_path, "pose", f"{use_len}_{compute_len}_{predict_len}_{stride_len}")

        self.use_len = use_len
        self.compute_len = compute_len
        self.predict_len = predict_len
        self.stride_len = stride_len
        self.fps = fps

        self.use_len_frames = int(use_len * fps)
        self.compute_len_frames = int(compute_len * fps)
        self.predict_len_frames = int(predict_len * fps)
        self.stride_len_frames = int(stride_len * fps)

        if not os.path.exists(self.imu_root_path) or not os.path.exists(self.pose_root_path):
            process(
                source_dir=source_dir,
                target_dir=root_path,
                smpl_model_dir=smpl_model_dir,
                use_len=use_len,
                compute_len=compute_len,
                predict_len=predict_len,
                stride_len=stride_len,
                fps=fps,
                subsets=subsets,
                device=device,
            )

        filenames = sorted(os.listdir(self.imu_root_path))
        self.imu_filepaths = [os.path.join(self.imu_root_path, f) for f in filenames]
        self.pose_filepaths = [os.path.join(self.pose_root_path, f) for f in filenames]

        self.cache = {}

    def __len__(self):
        return len(self.imu_filepaths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        imu = torch.tensor(np.load(self.imu_filepaths[idx]), dtype=torch.float32)   # (204, use+pred)
        pose = torch.tensor(np.load(self.pose_filepaths[idx]), dtype=torch.float32)  # (compute+pred, 24, 3)

        imu = torch.nan_to_num(imu, nan=0.0, posinf=0.0, neginf=0.0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)

        hist_imu = imu[:, :self.use_len_frames]
        current_pose = pose[:self.compute_len_frames]
        future_imu = imu[:, self.use_len_frames:]
        future_pose = pose[self.compute_len_frames:]

        self.cache[idx] = (hist_imu, current_pose, future_imu, future_pose)
        return self.cache[idx]


if __name__ == "__main__":
    dataset = AMASS_SYNTH_IMU(
        root_path="/mnt/mydata/yh/liming/workspace/future/mydata/AMASS_SYNTH_IMU_split",
        use_len=4,
        compute_len=1,
        predict_len=1,
        stride_len=1,
        fps=60,
        source_dir="/mnt/mydata/yh/liming/data/AMASS",
        smpl_model_dir="/mnt/mydata/yh/liming/workspace/future/SMPL",
        subsets=["CMU", "MPI_HDM05", "TotalCapture"],
        device="cuda:1" if torch.cuda.is_available() else "cpu",
    )

    print("数据集大小:", len(dataset))
    hist_imu, current_pose, future_imu, future_pose = dataset[0]
    print("历史 IMU shape:", hist_imu.shape)
    print("当前姿态 shape:", current_pose.shape)
    print("未来 IMU shape:", future_imu.shape)
    print("未来姿态 shape:", future_pose.shape)