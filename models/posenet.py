import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)  # [B, 512, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return x


def resnet(version, input_channels):
    if version == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], input_channels)
    elif version == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], input_channels)
    elif version == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], input_channels)
    return None


class PoseNet(nn.Module):
    def __init__(self, input_channels, resnet_verson, lstm_hidden, lstm_layers, lstm_dropout, target_time, target_poses, num_poses, num_keypoints=25, output_dim=2):
        super().__init__()
        self.resnet = resnet(resnet_verson, input_channels)
        if resnet_verson == "resnet18" or resnet_verson == "resnet34":
            self.lstm = nn.LSTM(input_channels + 512, lstm_hidden, lstm_layers, dropout=lstm_dropout, batch_first=True)
        elif resnet_verson == "resnet50":
            self.lstm = nn.LSTM(input_channels + 2048, lstm_hidden, lstm_layers, dropout=lstm_dropout, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden, input_channels)
        self.fc2 = nn.Linear(512, num_poses * num_keypoints * output_dim)
        self.fc3 = nn.Linear(512, target_poses * num_keypoints * output_dim)
        
        self.target_time = target_time
        self.target_poses = target_poses
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        features = self.resnet(x)  # [batch, 512]
        features = features.unsqueeze(1)  # [batch, 1, 512]
        now_pose = self.fc2(features)  # [B, 25*2]
        now_pose = now_pose.view(now_pose.size(0), self.num_poses, self.num_keypoints, self.output_dim)

        preds = []
        decoder_input = torch.zeros(x.size(0), 1, x.size(1), device=x.device)  # 初始输入为全零
        h, c = None, None
        # 自回归预测 future steps
        for _ in range(self.target_time):
            # 拼接当前输入 + 全局特征
            lstm_input = torch.cat([decoder_input, features], dim=-1)  # [batch, 1, imu_dim + 512]
            out, (h, c) = self.lstm(lstm_input, (h, c)) if h is not None else self.lstm(lstm_input)
            pred = self.fc1(out)  # [batch, 1, imu_dim]
            preds.append(pred)
            decoder_input = pred  # 下一步输入用当前预测

        future_x = torch.cat(preds, dim=1).permute(0, 2, 1)  # [batch, imu_dim, target_time]

        future_features = self.resnet(torch.cat([x[:, :, self.target_time:], future_x], dim=2))  # [batch, 512]
        future_features = future_features.unsqueeze(1)  # [batch, 1, 512]
        future_pose = self.fc3(future_features)  # [B, 25*2]
        future_pose = future_pose.view(future_pose.size(0), self.target_poses, self.num_keypoints, self.output_dim)

        # return now_pose, future_x, future_pose
        return torch.tanh(now_pose), future_x, torch.tanh(future_pose)


if __name__ == "__main__":
    x = torch.randn(32, 30, 150)
    model = PoseNet(input_channels=30, lstm_hidden=128, lstm_layers=2, target_time=50, target_poses=15, 
                    num_poses=15, num_keypoints=25, output_dim=2)
    now_pose, future_x, future_pose = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", now_pose.shape)
    print("输出形状:", future_x.shape)
    print("输出形状:", future_pose.shape)