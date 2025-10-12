import math
import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


# -------------------------
# replace BatchNorm1d -> GroupNorm (more stable for small batches)
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None, gn_groups=8):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        # use GroupNorm instead of BatchNorm for stability with small batches
        self.gn1 = nn.GroupNorm(num_groups=gn_groups if planes >= gn_groups else 1, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups if planes >= gn_groups else 1, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel, gn_groups=8):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, gn_groups=gn_groups)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, gn_groups=gn_groups)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, gn_groups=gn_groups)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, gn_groups=gn_groups)
        # final small conv
        self.conv4 = conv3x3(512, 512, stride=2)
        # no global pooling here because we need sequence length dimension
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, blocks, stride=1, group=1, gn_groups=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(num_groups=gn_groups if planes * block.expansion >= gn_groups else 1,
                             num_channels=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample, gn_groups=gn_groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group, gn_groups=gn_groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [B, C, L]
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c4  # [B, channels, L']


def resnet18(in_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel)


# -------------------------
# 1D sinusoidal positional encoding (deterministic, stable)
# -------------------------
class SinePositionalEncoding1D(nn.Module):
    def __init__(self, hidden_dim, temperature=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, length, device=None):
        """Return (length, hidden_dim) positional encodings"""
        if device is None:
            device = torch.device('cpu')
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, device=device).float() * -(math.log(self.temperature) / self.hidden_dim))
        pe = torch.zeros(length, self.hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L, hidden_dim]


# -------------------------
# Improved PoseDETR
# - smaller hidden dim
# - fewer transformer layers
# - LayerNorm in transformer (pre-norm), dropout
# - GroupNorm in backbone
# - deterministic positional encoding
# - initialize weights once via init_weights()
# - remove final sigmoid (let training target be normalized)
# -------------------------
class DeformableDETR(nn.Module):
    def __init__(self,
                 in_channels=30,
                 num_keypoints=25,
                 hidden_dim=128,            # reduce feature dim (easier to train)
                 nheads=4,                  # fewer heads
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dropout=0.1):
        super().__init__()
        # backbone
        self.backbone = resnet18(in_channel=in_channels)
        # project to transformer dim
        self.conv = nn.Conv1d(512, hidden_dim, 1)

        # positional encoding module (deterministic)
        self.pos_enc = SinePositionalEncoding1D(hidden_dim)

        # transformer: use LayerNorm (pre-norm) and dropout
        # note: set batch_first=False because we use [S, B, D] inputs
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False,  # 我们仍然使用 [S, B, D]
        )

        # queries
        self.query_pos = nn.Parameter(torch.randn(num_keypoints, hidden_dim))

        # head
        self.keypoint_head = nn.Linear(hidden_dim, 2)  # linear output; normalize labels accordingly during training

        # initialize weights once
        self._init_weights()

    def _init_weights(self):
        # initialize conv and linear layers properly (xavier)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                # defaults are fine
                pass
        # small init for query_pos
        nn.init.normal_(self.query_pos, mean=0., std=0.01)

    def forward(self, inputs):
        """
        inputs: Tensor [B, C, L]  (IMU sequence)
        returns: [B, num_keypoints, 2]  (linear outputs; training labels should be scaled appropriately)
        """
        B, C, L_in = inputs.shape

        # backbone -> [B, 512, L']
        x = self.backbone(inputs)
        # debug prints can be enabled if needed:
        # print('backbone out', x.shape)
        x = self.conv(x)  # [B, hidden_dim, L']
        L = x.shape[-1]

        # positional encoding (deterministic)
        pe = self.pos_enc(L, device=x.device)  # [L, hidden_dim]
        # transformer expects [S, B, D]
        src = x.permute(2, 0, 1) + pe.unsqueeze(1)  # [L, B, D]

        # expand queries to batch
        tgt = self.query_pos.unsqueeze(1).expand(-1, B, -1)  # [num_q, B, D]

        # pass through transformer
        hs = self.transformer(src, tgt)  # [num_q, B, D]

        # head: [num_q, B, D] -> [B, num_q, 2]
        out = self.keypoint_head(hs)  # linear outputs (no sigmoid)
        out = out.permute(1, 0, 2)  # [B, num_q, 2]
        return out


if __name__ == "__main__":
    model = DeformableDETR(in_channels=30, num_keypoints=25)
    model.eval()
    
    inputs = torch.randn(1, 30, 800)  # (batch=1, channel=3, length=800)
    with torch.no_grad():
        keypoints = model(inputs)
    
    print("输出形状:", keypoints.shape)  # [1, 10, 1]
    print("前5个关键点位置:", keypoints[0, :])