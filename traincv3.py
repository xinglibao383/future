import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloaderc_v2 import *
from utils.traincv3 import train as train
from models.resnet_mh import resnet18

# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/traincv3.py
if __name__ == "__main__":
    window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs, alpha, beta, gamma = 250, 25, 512, 0.15, 1e-3, 1e-4, 800, 1, 1, 1
    model = resnet18(30, 3, 16, 34)
    train_loader, val_loader = get_dataloaders_mc("/home/xinglibao/workspace/future/datamc/imu", window_size, stride, batch_size, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path='/home/xinglibao/workspace/future/outputs', timestamp=timestamp)
    logger.record(['Comment: multi classify'])
    logger.record([f'window_size = {window_size}, stride = {stride}, batch_size = {batch_size}, mask_ratio = {mask_ratio} '
                f'lr = {lr}, weight_decay = {weight_decay}, num_epochs = {num_epochs} '
                f'alpha = {alpha}, beta = {beta}, gamma = {gamma}'])
    train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, os.path.join('/data/xinglibao/outputs', timestamp), logger, alpha, beta, gamma)
