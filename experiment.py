import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloaderc import *
from utils.traincv2 import train as train
from models.mycnet import IMUPose
from models.mycnetv2 import IMUPose as IMUPosev2


def run(hidden_dim, num_layers, dropout, window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs, alpha, beta, gamma, logger):
    model = IMUPosev2(hidden_dim=hidden_dim, num_layers=num_layers, len_output=window_size/2/50, dropout=dropout)
    train_loader, val_loader = get_dataloaders("/data/xinglibao/data/future/imu", window_size, stride, batch_size, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    logger.record(["============================================================================================================================="])
    logger.record(["============================================================================================================================="])
    logger.record([f'Params: hidden_dim = {hidden_dim}, num_layers = {num_layers}, dropout = {dropout}, '
                f'window_size = {window_size}, stride = {stride}, batch_size = {batch_size}, mask_ratio = {mask_ratio} '
                f'lr = {lr}, weight_decay = {weight_decay}, num_epochs = {num_epochs} '
                f'alpha = {alpha}, beta = {beta}, gamma = {gamma} '])
    train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, '', logger, alpha, beta, gamma)


def experiment1():
    hidden_dim, num_layers, dropout, window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs, alpha, beta, gamma = 128, 2, 0.3, 250, 25, 512, 0.15, 1e-3, 1e-4, 800, 10, 0.001, 1
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path='/home/xinglibao/workspace/future/outputs', timestamp=timestamp, extend='experiment1')
    for hidden_dim in (64, 128, 256):
        for num_layers in (2, 3, 4):
            for dropout in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
                for window_size in (150, 200, 350, 300, 250):
                    for batch_size in (256, 512, 1024):
                        for mask_ratio in np.arange(0.0, 0.9, 0.15):
                            for lr in (0.01, 0.001, 0.0001):
                                for alpha in range(5, 100, 5):
                                    run(hidden_dim, num_layers, dropout, window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs, alpha, beta, gamma, logger)


if __name__ == "__main__":
    experiment1()