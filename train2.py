import os
import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train22 import train
from models.detr import *
from models.poselstm import *
from models.deformabledetr import *
from models.poseresnet import *


# devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
devices = [torch.device('cuda:2'), torch.device('cuda:3')]


# /home/xinglibao/anaconda3/envs/future/bin/python /home/xinglibao/workspace/future/train2.py
if __name__ == "__main__":
    mask_ratio, batch_size, lr, num_epochs = 0.25, 128, 1e-2, 800
    use_len, compute_len, predict_len, stride_len = 30, 15, 5, 15
    # model = PoseLSTM(input_size=30, hidden_size=512, num_layers=4, num_keypoints=25, output_size=2)
    # model = PoseLSTM(input_size=30, hidden_size=256, num_layers=2, num_keypoints=25, output_size=2)
    # model = DeformableDETR(in_channels=30, num_keypoints=25, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1)
    # model = DeformableDETR(in_channels=30, num_keypoints=25, hidden_dim=256, nheads=8, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1)
    model = poseresnet18(in_channel=30, num_poses=15)
    train_loader, val_loader = get_dataloaders_v2('/home/xinglibao/workspace/future/mydata', use_len, compute_len, predict_len, stride_len, batch_size, 0.8)
    output_save_path = os.path.join('/data/xinglibao/outputs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    train(model, train_loader, val_loader, lr, mask_ratio, num_epochs, devices, output_save_path, logger)
