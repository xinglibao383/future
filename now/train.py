import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train import train
from models.resnet import resnet18


if __name__ == "__main__":
    window_size, stride, batch_size, mask_ratio, lr, weight_decay, num_epochs, train_ratio = 150, 50, 512, 0.15, 1e-3, 1e-4, 800, 0.8
    model = resnet18(in_channel=30, num_classes=33)
    train_loader, val_loader = get_dataloaders("/data/xinglibao/data/future/now/imu", window_size, stride, batch_size, train_ratio)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path='/home/xinglibao/workspace/future/now/outputs', timestamp=timestamp)
    logger.record([f'Params: window_size = {window_size}, stride = {stride}, mask_ratio = {mask_ratio}, batch_size = {batch_size}, lr = {lr}, weight_decay = {weight_decay}, num_epochs = {num_epochs}'])
    train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, os.path.join('/data/xinglibao/outputs', timestamp), logger)
