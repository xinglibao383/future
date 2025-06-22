import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader_v2 import *
from utils.train_v2 import train
from models.mynet import IMUPose


if __name__ == "__main__":
    len_input, len_predict, lr, num_epochs = 4, 1, 1e-3, 400
    model = IMUPose(len_input, len_predict)
    train_loader, val_loader = get_dataloaders('/home/xinglibao/workspace/future/data', len_input, len_predict, 64, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    output_save_path = os.path.join('/home/xinglibao/workspace/future/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    logger = Logger(save_path=output_save_path)
    logger.record([f'Params: len_input = {len_input}, len_predict = {len_predict}, lr = {lr}, num_epochs = {num_epochs}'])
    train(model, train_loader, val_loader, lr, num_epochs, devices, output_save_path, logger)
