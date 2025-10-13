import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *


# devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
devices = [torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
# devices = [torch.device('cuda:2'), torch.device('cuda:3')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_save_path = os.path.join('/data/xinglibao/outputs', timestamp)
logger = Logger(save_path=output_save_path, timestamp=timestamp)

def train():
    mask_ratio, batch_size, lr, num_epochs = 0.25, 128, 1e-2, 800
    lstm_hidden, lstm_layers = 128, 2
    use_len, compute_len, predict_len, stride_len = 45, 15, 15, 15
    model = PoseNet(input_channels=30, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3('/home/xinglibao/workspace/future/mydata', use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    train3(model, train_loader, val_loader, mask_ratio, lr, num_epochs, devices, output_save_path, logger)


# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train3.py
if __name__ == "__main__":
    train()