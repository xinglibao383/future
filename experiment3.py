import os
import socket
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *
from utils.predict_max_predict_len import *
from itertools import combinations
from glob import glob


def already_done(target_tuple):
    if socket.gethostname() != "lenovo-Lenovo-WenTian-WA5480-G3":
        root_dir = "/root/future/outputs/experiment2028/exclude_device_200epoch"
    else:
        root_dir = "/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/exclude_device_200epoch"
    target_str = str(target_tuple)
    txt_files = glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if target_str in first_line:
                return True
        except Exception as e:
            print(f"读取文件失败: {txt_file}, 错误: {e}")
    return False


devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
devices = [torch.device('cuda:3')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if socket.gethostname() == "lenovo-Lenovo-WenTian-WA5480-G3":
    output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028'
    data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'
else:
    output_save_path = '/root/future/outputs/experiment2028'
    data_root_path = '/root/future/mydata'


def exclude_device_experiment(exclude_device_idx=None):
    global output_save_path
    this_output_save_path = os.path.join(output_save_path, "exclude_device_200epoch")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=this_output_save_path, timestamp=timestamp)
    logger.record([f'备注: 设备消融实验, exclude_device_idx = {exclude_device_idx}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "exclude_device_idx": exclude_device_idx, 
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30-6*len(exclude_device_idx), resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8, exclude_device_idx=exclude_device_idx)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, this_output_save_path, logger, timestamp)


def cross_environment_experiment(cross, cross_idx):
    global output_save_path
    this_output_save_path = os.path.join(output_save_path, cross)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=this_output_save_path, timestamp=timestamp)
    logger.record([f'备注: 跨域实验, cross={cross}, cross_idx={cross_idx}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "cross": cross, "cross_idx": cross_idx, 
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3_cross_experiment(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, cross, cross_idx)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, this_output_save_path, logger, timestamp)


def max_predict_len(checkpoint_filepath=None):
    global output_save_path
    output_save_path = os.path.join(output_save_path, "max_predict_len")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 最大预测长度实验'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "checkpoint_filepath": checkpoint_filepath,
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])

    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=50, target_poses=15, num_poses=compute_len, num_keypoints=25, output_dim=2)
    if checkpoint_filepath is not None:
        _, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, 150, stride_len, batch_size, 0.8, random_seed=3407)
        val_loss_mpjpe(model, checkpoint_filepath, devices[0], val_loader, logger)
    else:
        train_loader, val_loader = get_dataloaders_v3_max_predict_len(data_root_path, use_len, compute_len, 150, stride_len, batch_size, 0.8)
        train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


def ablation_mask(mask_ratio):
    global output_save_path
    this_output_save_path = os.path.join(output_save_path, "ablation_mask")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=this_output_save_path, timestamp=timestamp)
    logger.record([f'备注: MASK消融实验, mask_ratio={mask_ratio}'])
    batch_size, lr, num_epochs, loss_func = 512, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8, random_seed=3407)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, this_output_save_path, logger, timestamp)


def select_backbone(imu_generator):
    global output_save_path
    this_output_save_path = os.path.join(output_save_path, "select_backbone")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=this_output_save_path, timestamp=timestamp)
    logger.record([f'备注: 选择backbone, imu_generator={imu_generator}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson = "resnet18"
    mamba_d_state, mamba_d_conv, mamba_expand, mamba_dropout = 64, 4, 2, 0.1
    lstm_hidden, lstm_layers, lstm_dropout = 128, 2, 0.1
    gru_hidden, gru_layers, gru_dropout = 128, 2, 0.1
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "mamba_d_state": mamba_d_state, "mamba_d_conv": mamba_d_conv, "mamba_expand": mamba_expand, 
        "lstm_hidden": lstm_hidden, "lstm_layers": lstm_layers, "lstm_dropout": lstm_dropout,
        "gru_hidden": gru_hidden, "gru_layers": gru_layers, "gru_dropout": gru_dropout,
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    if imu_generator == "lstm":
        imu_generator_params = (lstm_hidden, lstm_layers, lstm_dropout)
    elif imu_generator == "transformer":
        imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    elif imu_generator == "gru":
        imu_generator_params = (gru_hidden, gru_layers, gru_dropout)
    elif imu_generator == "mamba":
        imu_generator_params = (mamba_d_state, mamba_d_conv, mamba_expand, mamba_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8, random_seed=3407)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, this_output_save_path, logger, timestamp)


# nohup /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment3.py > /dev/null 2>&1 &
# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment3.py
# nohup /usr/local/miniconda3/envs/future/bin/python /root/future/experiment3.py > /dev/null 2>&1 &
# /usr/local/miniconda3/envs/future/bin/python /root/future/experiment3.py
if __name__ == "__main__":
    for v in list(combinations([0, 1, 2, 3, 4], 4)):
        if not already_done(v):
            exclude_device_experiment(exclude_device_idx=v)

    # for idx in range(1, 4):
    #     cross_environment_experiment(cross="cross_environment", cross_idx=idx)

    # cross_environment_experiment(cross="cross_person", cross_idx=15)
    # for idx in range(16):
    #     cross_environment_experiment(cross="cross_person", cross_idx=idx)

    # max_predict_len(checkpoint_filepath="/mnt/mydata/yh/liming/workspace/future/outputsnew/experiment/max_predict_len/20251020163457/checkpoints/epoch_199.pth")
    
    # for mr in [round(i * 0.05, 2) for i in range(1, 10)]:
    #     ablation_mask(mask_ratio=mr)

    # select_backbone(imu_generator="mamba")