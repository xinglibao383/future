import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.resnet import *
from models.generator.lstm import *
from models.generator.transformer import *


class PoseNet(nn.Module):
    def __init__(self, input_channels, resnet_verson, imu_generator, imu_generator_params, target_time, target_poses, num_poses, num_keypoints=25, output_dim=2):
        super().__init__()
        self.resnet = resnet(resnet_verson, input_channels)
        if resnet_verson == "resnet18" or resnet_verson == "resnet34":
            resent_feature_dim = 512
        elif resnet_verson == "resnet50":
            resent_feature_dim = 2048
        
        if imu_generator == "lstm":
            lstm_hidden, lstm_layers, lstm_dropout = imu_generator_params
            self.imu_predictor = LSTMGenerator(
                input_dim=resent_feature_dim, 
                hidden_dim=lstm_hidden, 
                output_dim=input_channels, 
                target_len=target_time, 
                num_layers=lstm_layers,
                dropout=lstm_dropout
            )
        elif imu_generator == "transformer":
            transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = imu_generator_params
            self.imu_predictor = TransformerGenerator(
                input_dim=resent_feature_dim, 
                hidden_dim=transformer_hidden, 
                output_dim=input_channels, 
                num_layers=transformer_layers, 
                nhead=transformer_nhead, 
                dropout=transformer_dropout, 
                target_len=target_time
            )

        self.fc1 = nn.Linear(resent_feature_dim, num_poses * num_keypoints * output_dim)
        self.fc2 = nn.Linear(resent_feature_dim, target_poses * num_keypoints * output_dim)
        
        self.target_time = target_time
        self.target_poses = target_poses
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        features = self.resnet(x)  # [batch, 512]
        features = features.unsqueeze(1)  # [batch, 1, 512]
        now_pose = self.fc1(features)  # [B, 25*2]
        now_pose = now_pose.view(now_pose.size(0), self.num_poses, self.num_keypoints, self.output_dim)

        future_x = self.imu_predictor(features)  # [batch, imu_dim, target_time]

        future_features = self.resnet(torch.cat([x[:, :, self.target_time:], future_x], dim=2))  # [batch, 512]
        future_features = future_features.unsqueeze(1)  # [batch, 1, 512]
        future_pose = self.fc2(future_features)  # [B, 25*2]
        future_pose = future_pose.view(future_pose.size(0), self.target_poses, self.num_keypoints, self.output_dim)

        return torch.tanh(now_pose), future_x, torch.tanh(future_pose)


if __name__ == "__main__":
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 128, 1e-2, 800, "l1"
    resnet_verson, lstm_hidden, lstm_layers, lstm_dropout = "resnet18", 128, 2, 0
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    
    x = torch.randn(32, 30, 150)
    now_pose, future_x, future_pose = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", now_pose.shape)
    print("输出形状:", future_x.shape)
    print("输出形状:", future_pose.shape)