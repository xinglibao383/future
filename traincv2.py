import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloaderc import *
from utils.traincv2 import train
from models.mycnet import IMUPose
from models.mycnetv2 import IMUPose as IMUPosev2


if __name__ == "__main__":
    hidden_dim, num_layers, dropout, window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs = 128, 2, 0.3, 150, 50, 512, 0.25, 1e-3, 1e-4, 800
    model = IMUPosev2(hidden_dim=hidden_dim, num_layers=num_layers, len_output=window_size/2/50, dropout=dropout)
    train_loader, val_loader = get_dataloaders("/home/xinglibao/workspace/future/datac/imu", window_size, stride, batch_size, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    output_save_path = os.path.join('/home/xinglibao/workspace/future/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    logger = Logger(save_path=output_save_path)
    logger.record([f'Params: hidden_dim = {hidden_dim}, num_layers = {num_layers}, dropout = {dropout}, '
                f'window_size = {window_size}, stride = {stride}, batch_size = {batch_size}, mask_ratio = {mask_ratio} '
                f'lr = {lr}, weight_decay = {weight_decay}, num_epochs = {num_epochs}'])
    train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, output_save_path, logger)
