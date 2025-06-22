import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloaderc import *
from utils.trainc import train
from models.resnetc import resnet18


if __name__ == "__main__":
    window_size, stride, lr, num_epochs = 100, 70, 1e-3, 400
    model = resnet18(in_channel=5*6, num_classes=34)
    train_loader, val_loader = get_dataloaders("/home/xinglibao/workspace/future/datac/imu", window_size, stride, 64, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    output_save_path = os.path.join('/home/xinglibao/workspace/future/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    logger = Logger(save_path=output_save_path)
    logger.record([f'Params: window_size = {window_size}, stride = {stride}, lr = {lr}, num_epochs = {num_epochs}'])
    train(model, train_loader, val_loader, lr, num_epochs, devices, output_save_path, logger)
