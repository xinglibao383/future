import torch
import torch.nn as nn
from models.mynet import IMUPose
import numpy as np
from utils.dataloader_v2 import get_dataloaders


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders('/home/xinglibao/workspace/future/data', 4, 1, 64, 0.8)
    for i, (x_imu, y_imu, x_pose, y_pose) in enumerate(train_loader):
        print(i, x_imu.shape, y_imu.shape, x_pose.shape, y_pose.shape)

    len_input, len_output = 4, 1
    x = torch.randn(32, len_input*50, 6)
    model = IMUPose(len_input, len_output)
    pose, future_pose1, future_imu, future_pose2 = model(x)
    print(pose.shape, future_pose1.shape, future_imu.shape, future_pose2.shape)