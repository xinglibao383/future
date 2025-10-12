import torch
from torch import nn


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
        return c4


def resnet18(in_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel)


class PoseDETR(nn.Module):
    def __init__(self, in_channels=3, num_keypoints=10, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # 替换为 1D ResNet backbone
        self.backbone = resnet18(in_channel=in_channels)
        self.conv = nn.Conv1d(512, hidden_dim, 1)

        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nheads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers
        )

        # 1D 位置编码
        self.pos_embed = nn.Parameter(torch.rand(500, hidden_dim))  # 假设序列长度<=500
        self.query_pos = nn.Parameter(torch.rand(num_keypoints, hidden_dim))

        # 输出关键点 (x)，可以改为2表示(x, y)
        self.keypoint_head = nn.Linear(hidden_dim, 2)

    def forward(self, inputs):
        # inputs: [B, C, L]
        x = self.backbone(inputs)
        print(x.shape)
        x = self.conv(x)  # [B, hidden_dim, L']
        L = x.shape[-1]

        # 添加位置编码
        src = x.permute(2, 0, 1) + self.pos_embed[:L].unsqueeze(1)  # [L, B, hidden_dim]

        hs = self.transformer(src, self.query_pos.unsqueeze(1))  # [num_kp, B, hidden_dim]

        keypoints = self.keypoint_head(hs).sigmoid()  # [num_kp, B, 1]
        keypoints = keypoints.permute(1, 0, 2)  # [B, num_kp, 1]
        return keypoints



if __name__ == "__main__":
    model = PoseDETR(in_channels=30, num_keypoints=25)
    model.eval()
    
    inputs = torch.randn(1, 30, 800)  # (batch=1, channel=3, length=800)
    with torch.no_grad():
        keypoints = model(inputs)
    
    print("输出形状:", keypoints.shape)  # [1, 10, 1]
    print("前5个关键点位置:", keypoints[0, :])