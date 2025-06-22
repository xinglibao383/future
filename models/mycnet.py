import torch
import torch.nn as nn


class MLPC(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_classes, dropout=0.3):
        super(MLPC, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.decoder(x)
    

class MLP(nn.Module):
    def __init__(self, feature_dim, hidden_size, output_len, output_dim, dropout=0.3):
        super(MLP, self).__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_len * output_dim)
        )

    def forward(self, x):
        out = self.decoder(x)
        out = out.view(-1, self.output_dim, self.output_len)
        return out


def conv3x3(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


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
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        output = self.avg_pool(c4)
        return output.view(output.size(0), -1)


def resnet18(in_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel)

class IMUPose(nn.Module):
    def __init__(self, len_output):
        super(IMUPose, self).__init__()
        self.imu_channel = 5 * 6
        self.imu_feature_extractor = resnet18(in_channel=self.imu_channel)
        self.imu_classfier = MLPC(512, 256, 34, 0.3)
        self.imu_predictor = MLP(feature_dim=512, hidden_size=256, output_len=int(len_output*50), output_dim=self.imu_channel)

    def forward(self, imu):
        imu_feature = self.imu_feature_extractor(imu)
        label1 = self.imu_classfier(imu_feature)
        imu2 = self.imu_predictor(imu_feature)
        # todo 可以把之前的imu拼一些给imu2
        imu2_feature = self.imu_feature_extractor(imu2)
        label2 = self.imu_classfier(imu2_feature)
        return label1, imu2, label2

if __name__ == "__main__":
    len_input = 1.5
    x = torch.randn(32, 30, int(len_input*50))
    model = IMUPose(len_input)
    label1, imu2, label2 = model(x)
    print(label1.shape, imu2.shape, label2.shape)