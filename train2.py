import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train22 import train
from models.detr import *
from models.poselstm import *


if __name__ == "__main__":
    batch_size, lr, num_epochs = 512, 1e-2, 4000000
    model = PoseLSTM(input_size=30, hidden_size=256, num_layers=2, num_keypoints=25, output_size=2)
    model = PoseLSTM(input_size=30, hidden_size=512, num_layers=4, num_keypoints=25, output_size=2)
    model = PoseDETR(in_channels=30, num_keypoints=25)
    train_loader, val_loader = get_dataloaders_v2('/home/xinglibao/workspace/future/mydata', batch_size, 0.8)
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    # devices = [torch.device('cuda:2'), torch.device('cuda:3')]
    output_save_path = os.path.join('/home/xinglibao/workspace/future/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    train(model, train_loader, val_loader, lr, num_epochs, devices, output_save_path, logger)
