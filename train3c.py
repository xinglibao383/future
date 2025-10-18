import os
import shutil
import torch
import torch.nn as nn
from datetime import datetime
from utils.logger import Logger
from classify.utils.xrfv22c import get_dataloaders as get_dataloaders_c
from classify.utils.train_rebuild import train as train_rebuild
from classify.models.mamba import *


devices = [torch.device('cuda:0'), torch.device('cuda:1')]
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputsnewc'
data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydatac'
logger = Logger(save_path=output_save_path, timestamp=timestamp)
positive_labels = [2, 11, 17, 19, 21, 22]


def train():
    logger.record([f'备注: 使用mamba测试对指定类别的重构效果'])
    mask_ratio, batch_size, lr, weight_decay, num_epochs, loss_func = 0.25, 256, 1e-4, 1e-4, 200, "l1"
    mamba_d_state, mamba_d_conv, mamba_expand = 16, 4, 2
    window_size, stride = 30, 15
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "epochs": num_epochs, "loss_func": loss_func,
        "mamba_d_state": mamba_d_state, "mamba_d_conv": mamba_d_conv, "mamba_expand": mamba_expand, 
        "window_size": window_size, "stride": stride, 
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    model = MambaGenerator(input_dim=30, d_state=64, d_conv=4, expand=2)
    train_loader, val_loader = get_dataloaders_c(data_root_path, window_size, stride, positive_labels, batch_size, 0.8)
    train_rebuild(model, train_loader, val_loader, loss_func, mask_ratio, lr, weight_decay, num_epochs, devices, output_save_path, logger, timestamp)


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/train3c.py
if __name__ == "__main__":
    train()