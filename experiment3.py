import os
import shutil
import datetime
import torch
import torch.nn as nn
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *


# devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# output_save_path = '/data/xinglibao/outputs'
# data_root_path = '/home/xinglibao/workspace/future/mydata'
output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputsnew/experiment'
data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'


def exclude_device_experiment(exclude_device_idx=None):
    global output_save_path
    output_save_path = os.path.join(output_save_path, "exclude_device")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 设备消融实验, exclude_device_idx = {exclude_device_idx}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 300, "l1"
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
    model = PoseNet(input_channels=24, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8, exclude_device_idx=exclude_device_idx)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


def cross_environment_experiment():
    output_save_path = os.path.join(output_save_path, "cross_environment")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 跨域实验, 跨环境'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 300, "l1"
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
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment3.py
# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train3.py
if __name__ == "__main__":
    exclude_device_experiment(exclude_device_idx=2)