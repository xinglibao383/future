import torch
import torch.nn as nn
from models.resnet import resnet18
from models.mlp import MLP

class IMUPose(nn.Module):
    def __init__(self, len_input, len_output):
        super(IMUPose, self).__init__()
        self.imu_feature_extractor = resnet18(in_channel=6)
        self.pose_predictor = MLP(feature_dim=512, hidden_size=256, output_len=len_input*15, output_dim=25*2)
        self.pose_feature_extractor = resnet18(in_channel=25*2)
        self.future_pose_predictor = MLP(feature_dim=512, hidden_size=256, output_len=len_output*15, output_dim=25*2)
        self.imu_predictor = MLP(feature_dim=512, hidden_size=256, output_len=len_output*50, output_dim=6)

    def forward(self, imu):
        imu_feature = self.imu_feature_extractor(imu)
        pose = self.pose_predictor(imu_feature)
        pose_feature = self.pose_feature_extractor(pose)
        future_pose1 = self.future_pose_predictor(pose_feature)
        future_imu = self.imu_predictor(imu_feature)
        future_imu_feature = self.imu_feature_extractor(future_imu)
        future_pose2 = self.pose_predictor(future_imu_feature)
        return pose, future_pose1, future_imu, future_pose2

if __name__ == "__main__":
    len_input, len_output = 4, 1
    x = torch.randn(32, len_input*50, 6)
    model = IMUPose(len_input, len_output)
    pose, future_pose1, future_imu, future_pose2 = model(x)
    print(pose.shape, future_pose1.shape, future_imu.shape, future_pose2.shape)