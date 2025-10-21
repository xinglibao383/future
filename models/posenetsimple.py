import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.resnet import *
from models.generator.transformer import *


class PoseNetSimple(nn.Module):
    def __init__(self, target_poses, num_keypoints=25, output_dim=2):
        super().__init__()
        self.resnet = resnet("resnet18", 30)
        self.fc = nn.Linear(512, target_poses * num_keypoints * output_dim)
        self.target_poses = target_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        features = self.resnet(x)           # [batch, 512]
        features = features.unsqueeze(1)    # [batch, 1, 512]
        poses = self.fc(features)           # [B, 25*2]
        return torch.tanh(poses.view(poses.size(0), self.target_poses, self.num_keypoints, self.output_dim))