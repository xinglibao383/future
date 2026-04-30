import datetime
import torch
import socket
from utils.logger import Logger
from utils.dataloader import *
from utils.train_baseline_dataset import train
from models.comparison.baselines24 import build_baseline_model
from utils.train3_dataset import train as train3
from models.posenet24 import *


torch.manual_seed(3407)
devices = [torch.device('cuda:1')]
DATASET = 'AMASS'
# DATASET = 'DIP-IMU'
# DATASET = 'IMUPoser'
output_save_path = os.path.join('/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/dataset', DATASET)


def ours():
    global output_save_path
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 数据集={DATASET}, baseline=aipose'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    compute_len, predict_len = 25 if DATASET == 'IMUPoser' else 60, 25 if DATASET == 'IMUPoser' else 60
    input_channels, need_normalize, alpha, beta, gamma = 60 if DATASET == 'IMUPoser' else 204, True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=input_channels, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 60 * 60), target_poses=predict_len, num_poses=compute_len, num_keypoints=24, output_dim=3)
    if DATASET == 'DIP-IMU':
        train_loader, val_loader = get_dataloaders_dip_imu(4, 1, 1, 1, batch_size, 0.8, random_seed=3407)
    elif DATASET == 'AMASS':   
        train_loader, val_loader = get_dataloaders_amass(4, 1, 1, 1, batch_size, 0.8, random_seed=3407)
    else:
        train_loader, val_loader = get_dataloaders_imuposer(4, 1, 1, 1, batch_size, 0.8, random_seed=3407)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


def experiment_baseline(baseline):
    global output_save_path
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 数据集={DATASET}, baseline={baseline}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, hidden_dim, num_layers, nhead, dropout = "resnet18", 128, 2, 4, 0.1
    compute_len = 25 if DATASET == 'IMUPoser' else 60
    input_channels, train_ratio, need_normalize = 60 if DATASET == 'IMUPoser' else 204, 0.8, True
    params = {
        "baseline": baseline, "resnet_verson": resnet_verson,
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "nhead": nhead, "dropout": dropout,
        "input_channels": input_channels, "train_ratio": train_ratio, 
        "need_normalize": need_normalize,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    model = build_baseline_model(name=baseline, input_channels=input_channels, num_poses=compute_len, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout, resnet_verson=resnet_verson, num_keypoints=24, output_dim=3)
    if DATASET == 'DIP-IMU':
        train_loader, val_loader = get_dataloaders_dip_imu(4, 1, 1, 1, batch_size, train_ratio, random_seed=3407)
    elif DATASET == 'AMASS':   
        train_loader, val_loader = get_dataloaders_amass(4, 1, 1, 1, batch_size, train_ratio, random_seed=3407)
    else:
        train_loader, val_loader = get_dataloaders_imuposer(4, 1, 1, 1, batch_size, train_ratio, random_seed=3407)
    train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, num_epochs, devices, output_save_path, logger, timestamp)


# nohup /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_dataset.py > /dev/null 2>&1 &
# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_dataset.py
if __name__ == "__main__":
    ours()
    # experiment_baseline(baseline="pip_like_recon")
    # experiment_baseline(baseline="asip_like_recon")
    # experiment_baseline(baseline="mobileposer_like_recon")
    # experiment_baseline(baseline="imuposer_like_recon")
    # experiment_baseline(baseline="tip_like_recon")
    # experiment_baseline(baseline="dynaip_like_recon")