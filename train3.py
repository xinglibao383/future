import os
import shutil
import datetime
import torch
import torch.nn as nn
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *


def clean_outputs(root_dir="/mnt/mydata/yh/liming/workspace/future/outputsnew", min_epoch=15):
    min_lines = 2 + 2 * min_epoch
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        txt_file_path = os.path.join(folder_path, f"{folder_name}.txt")
        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < min_lines:
            print(f"删除文件夹 {folder_path}，因为行数只有 {len(lines)}")
            shutil.rmtree(folder_path)


# devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# output_save_path = '/data/xinglibao/outputs'
# data_root_path = '/home/xinglibao/workspace/future/mydata'
output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputsnew'
data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'
logger = Logger(save_path=output_save_path, timestamp=timestamp)


def train():
    logger.record([f'备注: 使用场景1、场景2、场景3数据, 对transformer调参'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 300, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    lstm_hidden, lstm_layers, lstm_dropout = 128, 2, 0.1
    gru_hidden, gru_layers, gru_dropout = 128, 2, 0.1
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    # transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 256, 4, 8, 0.2
    # transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 512, 6, 8, 0.3
    use_len, compute_len, predict_len, stride_len = 45, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
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

    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/train3.py
# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train3.py
if __name__ == "__main__":
    train()