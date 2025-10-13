import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train22 import train
from models.poseresnet import *
from models.detr import *


# devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
devices = [torch.device('cuda:2'), torch.device('cuda:3')]


# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train2.py
if __name__ == "__main__":
    mask_ratio, batch_size, lr, num_epochs = 0.25, 128, 1e-2, 800
    use_len, compute_len, predict_len, stride_len = 30, 15, 5, 15
    model = poseresnet18(in_channel=30, num_poses=15)
    model = DETR(num_queries=compute_len, num_keypoints=25, dim_keypoint=2, input_channels=30, lstm_layers=4, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    train_loader, val_loader = get_dataloaders_v2('/home/xinglibao/workspace/future/mydata', use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    output_save_path = os.path.join('/data/xinglibao/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    train(model, train_loader, val_loader, mask_ratio, lr, num_epochs, devices, output_save_path, logger)
