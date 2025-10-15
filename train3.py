import os
import shutil
import datetime
import torch
import torch.nn as nn
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *


def clean_outputs(root_dir="/data/xinglibao/outputs", min_epoch=15):
    min_lines = 2 + 2 * min_epoch
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        txt_file_path = os.path.join(folder_path, f"{folder_name}.txt")
        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < min_lines:
            print(f"删除文件夹 {folder_path}，因为行数只有 {len(lines)}")
            shutil.rmtree(folder_path)


devices = [torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3'), torch.device('cuda:0')]
# devices = [torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
# devices = [torch.device('cuda:2'), torch.device('cuda:3')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_save_path = '/data/xinglibao/outputs'
logger = Logger(save_path=output_save_path, timestamp=timestamp)


def train():
    logger.record([f'备注: 看看哪个场景的数据比较差, 使用场景1数据, 并且对imu和pose进行归一化'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 128, 1e-2, 800, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    lstm_hidden, lstm_layers, lstm_dropout = 128, 2, 0
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0
    use_len, compute_len, predict_len, stride_len = 45, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, 
        "lstm_hidden": lstm_hidden, "lstm_layers": lstm_layers, "lstm_dropout": lstm_dropout,
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])

    if imu_generator == "lstm":
        imu_generator_params = (lstm_hidden, lstm_layers, lstm_dropout)
    elif imu_generator == "transformer":
        imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)

    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3('/home/xinglibao/workspace/future/mydata', use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train3.py
if __name__ == "__main__":
    train()