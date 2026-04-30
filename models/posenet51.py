import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.resnet import *
from models.generator.lstm import *
from models.generator.transformer import *
from models.generator.gru import *
from models.generator.mamba import *


class PoseNet51(nn.Module):
    def __init__(self, input_channels, resnet_verson, pose_generator_params, target_poses, num_poses, num_keypoints=24, output_dim=3):
        super().__init__()
        self.resnet = resnet(resnet_verson, input_channels)
        resent_feature_dim = 512
        transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = pose_generator_params
        self.fc = nn.Linear(resent_feature_dim, num_poses * num_keypoints * output_dim)
        self.pose_predictor = TransformerPoseGenerator(
            num_keypoints=25, 
            input_dim=2, 
            hidden_dim=transformer_hidden, 
            num_layers=transformer_layers, 
            nhead=transformer_nhead, 
            dropout=transformer_dropout, 
            target_len=target_poses,
        )
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        features = self.resnet(x)           # [batch, 512]
        features = features.unsqueeze(1)    # [batch, 1, 512]
        now_pose = self.fc(features)
        now_pose = now_pose.view(now_pose.size(0), self.num_poses, self.num_keypoints, self.output_dim)
        future_pose = self.pose_predictor(now_pose)
        return now_pose, None, future_pose