import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
# from utils.train import train
from utils.train_with_confidence import train
from models.lstm import LSTM


if __name__ == "__main__":
    # len_input, len_predict, hidden_size, num_layers, lr, num_epochs = 30, 15, 512, 3, 1e-3, 400
    # len_input, len_predict, hidden_size, num_layers, lr, num_epochs = 45, 15, 512, 3, 1e-3, 400
    len_input, len_predict, hidden_size, num_layers, lr, num_epochs = 60, 15, 512, 3, 1e-3, 400
    # len_input, len_predict, hidden_size, num_layers, lr, num_epochs = 75, 15, 512, 3, 1e-3, 400
    model = PoseDETR(in_channels=30, num_keypoints=25)
    train_loader, val_loader = get_dataloaders('/home/xinglibao/workspace/future/data', len_input, len_predict, 64, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    devices = [torch.device('cuda:2'), torch.device('cuda:3')]
    output_save_path = os.path.join('/home/xinglibao/workspace/future/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'Params: len_input = {len_input}, len_predict = {len_predict}, hidden_size = {hidden_size}, num_layers = {num_layers}, lr = {lr}, num_epochs = {num_epochs}'])
    train(model, train_loader, val_loader, lr, num_epochs, devices, output_save_path, logger)
