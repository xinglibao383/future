import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import resnet
from models.posenet import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalProjectionHead(nn.Module):
    def __init__(self, hidden_dim, num_poses, num_keypoints=25, output_dim=2, dropout=0.0):
        super().__init__()
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim
        pose_dim = num_keypoints * output_dim
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, seq_features):
        pooled = F.adaptive_avg_pool1d(seq_features.transpose(1, 2), self.num_poses).transpose(1, 2)
        poses = self.out(pooled)
        poses = poses.view(poses.size(0), self.num_poses, self.num_keypoints, self.output_dim)
        return torch.tanh(poses)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, memory):
        attn_out, _ = self.attn(query, memory, memory, need_weights=False)
        x = self.norm1(query + attn_out)
        return self.norm2(x + self.ffn(x))


class QueryPoseHead(nn.Module):
    def __init__(self, hidden_dim, num_poses, num_keypoints=25, output_dim=2, nhead=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_poses, hidden_dim))
        self.query_pos = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max(512, num_poses + 8))
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_keypoints * output_dim),
        )
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, memory):
        q = self.query.expand(memory.size(0), -1, -1)
        q = self.query_pos(q)
        for block in self.blocks:
            q = block(q, memory)
        out = self.proj(q)
        out = out.view(memory.size(0), self.num_poses, self.num_keypoints, self.output_dim)
        return torch.tanh(out)


class SequenceStructureModule(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.local = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.global_conv = nn.Conv1d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.smooth = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.gate = nn.Sequential(
            nn.Conv1d(dim * 3, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        xt = x.transpose(1, 2)
        local_feat = self.local(xt)
        global_feat = self.global_conv(xt)
        smooth_feat = self.smooth(xt)
        gate = self.gate(torch.cat([local_feat, global_feat, smooth_feat], dim=1))
        fused = local_feat + gate * global_feat + (1.0 - gate) * smooth_feat
        return (self.out(fused) + xt).transpose(1, 2)


class TemporalSmoothModulation(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        xt = x.transpose(1, 2)
        smooth = self.pointwise(self.depthwise(xt))
        gate = self.gate(xt)
        return (xt + self.dropout(gate * smooth)).transpose(1, 2)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0, bidirectional=True):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        seq = self.in_proj(x.transpose(1, 2))
        seq, _ = self.rnn(seq)
        return self.norm(self.out_proj(seq))


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=4, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(depth):
            dilation = 2 ** (i % 3)
            blocks.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*blocks)
        self.out_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        feat = self.in_proj(x)
        feat = self.out_norm(feat + self.net(feat))
        return feat.transpose(1, 2)


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, nhead=4, dropout=0.0, use_ssm=False):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout)
        self.ssm = SequenceStructureModule(hidden_dim, dropout=dropout) if use_ssm else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        seq = self.in_proj(x.transpose(1, 2))
        seq = self.pos(seq)
        if self.ssm is not None:
            seq = self.ssm(seq)
        return self.norm(self.encoder(seq))


class DeviceAwareEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, sensor_dropout=0.15, use_transformer=False, dropout=0.0):
        super().__init__()
        self.input_channels = input_channels
        self.num_devices = max(1, input_channels // 6)
        self.sensor_dim = max(1, input_channels // self.num_devices)
        self.proj = nn.Linear(self.sensor_dim, hidden_dim)
        self.device_embed = nn.Parameter(torch.randn(1, self.num_devices, 1, hidden_dim))
        self.sensor_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sensor_dropout = sensor_dropout
        if use_transformer:
            self.temporal = TransformerBackbone(hidden_dim, hidden_dim, num_layers=2, nhead=4, dropout=dropout,
                                                use_ssm=False)
            self.fallback = TransformerBackbone(input_channels, hidden_dim, num_layers=2, nhead=4, dropout=dropout,
                                                use_ssm=False)
        else:
            self.temporal = GRUEncoder(hidden_dim, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True)
            self.fallback = GRUEncoder(input_channels, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        b, c, t = x.shape
        if c % self.num_devices != 0:
            return self.fallback(x)
        xs = x.view(b, self.num_devices, self.sensor_dim, t).permute(0, 1, 3, 2)
        feat = self.proj(xs) + self.device_embed[:, :self.num_devices]
        if self.training and self.sensor_dropout > 0:
            keep_mask = (torch.rand(b, self.num_devices, 1, 1, device=x.device) > self.sensor_dropout).float()
            if keep_mask.sum(dim=1).min() == 0:
                keep_mask[:, 0] = 1.0
            feat = feat * keep_mask
        score = self.sensor_score(feat).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (feat * weight).sum(dim=1)
        return self.norm(self.temporal(pooled.transpose(1, 2)))


class BaseReconstructor(nn.Module):
    def __init__(self, num_poses, num_keypoints=25, output_dim=2, hidden_dim=192, dropout=0.1):
        super().__init__()
        self.num_poses = num_poses
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pose_head = TemporalProjectionHead(hidden_dim, num_poses, num_keypoints, output_dim, dropout=dropout)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, seq_features):
        return self.pose_head(seq_features)

    def forward(self, x):
        return self.decode(self.encode(x))


class PIPLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_poses=15, num_keypoints=25, output_dim=2, dropout=0.1,
                 **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.stage1 = GRUEncoder(input_channels, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True)
        self.stage2 = GRUEncoder(input_channels + hidden_dim, hidden_dim, num_layers=2, dropout=dropout,
                                 bidirectional=True)
        self.stage3 = GRUEncoder(input_channels + hidden_dim, hidden_dim, num_layers=2, dropout=dropout,
                                 bidirectional=True)
        self.motion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def temporal_difference(self, x):
        diff = x[:, :, 1:] - x[:, :, :-1]
        return F.pad(diff, (1, 0))

    def encode(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(torch.cat([x, s1.transpose(1, 2)], dim=1))
        dx = self.temporal_difference(x)
        s3 = self.stage3(torch.cat([dx, s2.transpose(1, 2)], dim=1))
        gated_s3 = self.motion_gate(s3) * s3
        return self.norm(self.fuse(torch.cat([s1, s2, gated_s3], dim=-1)))


class TIPLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_layers=3, nhead=4, num_poses=15, num_keypoints=25,
                 output_dim=2, dropout=0.1, **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.backbone = TransformerBackbone(input_channels, hidden_dim, num_layers=num_layers, nhead=nhead,
                                            dropout=dropout, use_ssm=False)
        self.memory_refine = TemporalSmoothModulation(hidden_dim, dropout=dropout)
        self.query_head = QueryPoseHead(hidden_dim, num_poses, num_keypoints, output_dim, nhead=nhead, num_layers=2,
                                        dropout=dropout)

    def encode(self, x):
        return self.memory_refine(self.backbone(x))

    def decode(self, seq_features):
        return self.query_head(seq_features)


class IMUPoserLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_poses=15, num_keypoints=25, output_dim=2, dropout=0.1,
                 **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.device_encoder = DeviceAwareEncoder(input_channels, hidden_dim, sensor_dropout=0.2, use_transformer=False,
                                                 dropout=dropout)
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def encode(self, x):
        seq = self.device_encoder(x)
        return self.norm(seq + self.refine(seq))


class DynaIPLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_poses=15, num_keypoints=25, output_dim=2, dropout=0.1,
                 **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.input_channels = input_channels
        self.num_devices = max(1, input_channels // 6)
        self.sensor_dim = max(1, input_channels // self.num_devices)
        self.device_groups = self.build_device_groups(self.num_devices)
        self.part_raw = nn.ModuleList()
        self.part_dyn = nn.ModuleList()
        self.part_fuse = nn.ModuleList()
        for group in self.device_groups:
            group_channels = len(group) * self.sensor_dim
            self.part_raw.append(
                GRUEncoder(group_channels, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True))
            self.part_dyn.append(
                GRUEncoder(group_channels, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True))
            self.part_fuse.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.global_fuse = nn.Sequential(
            nn.Linear(hidden_dim * len(self.device_groups), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def build_device_groups(self, num_devices):
        if num_devices <= 1:
            return [[0]]
        if num_devices == 2:
            return [[0], [1]]
        if num_devices == 3:
            return [[0], [1], [2]]
        if num_devices == 4:
            return [[0], [1, 2], [3]]
        return [[0, 1], [2], list(range(3, num_devices))]

    def temporal_difference(self, x):
        diff = x[:, :, 1:] - x[:, :, :-1]
        return F.pad(diff, (1, 0))

    def select_group(self, x, group):
        chunks = []
        for idx in group:
            start, end = idx * self.sensor_dim, (idx + 1) * self.sensor_dim
            chunks.append(x[:, start:end, :])
        return torch.cat(chunks, dim=1)

    def encode(self, x):
        part_features = []
        dx = self.temporal_difference(x)
        for group, raw_encoder, dyn_encoder, fuse in zip(self.device_groups, self.part_raw, self.part_dyn,
                                                         self.part_fuse):
            xg = self.select_group(x, group)
            dg = self.select_group(dx, group)
            raw_feat = raw_encoder(xg)
            dyn_feat = dyn_encoder(dg)
            part_features.append(fuse(torch.cat([raw_feat, dyn_feat], dim=-1)))
        seq = self.global_fuse(torch.cat(part_features, dim=-1))
        return self.norm(seq)


class ASIPLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_layers=3, nhead=4, num_poses=15, num_keypoints=25,
                 output_dim=2, dropout=0.1, **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout)
        self.ssm = SequenceStructureModule(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_mod = TemporalSmoothModulation(hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode(self, x):
        seq = self.input_proj(x.transpose(1, 2))
        seq = self.pos(seq)
        seq = self.ssm(seq)
        seq = self.encoder(seq)
        seq = self.temporal_mod(seq)
        return self.norm(seq)


class MobilePoserLikeReconstructor(BaseReconstructor):
    def __init__(self, input_channels, hidden_dim=192, num_poses=15, num_keypoints=25, output_dim=2, dropout=0.1,
                 **kwargs):
        super().__init__(num_poses, num_keypoints, output_dim, hidden_dim, dropout)
        self.device_encoder = DeviceAwareEncoder(input_channels, hidden_dim, sensor_dropout=0.25, use_transformer=True,
                                                 dropout=dropout)
        self.temporal_conv = TemporalConvEncoder(hidden_dim, hidden_dim, depth=4, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def encode(self, x):
        seq1 = self.device_encoder(x)
        seq2 = self.temporal_conv(seq1.transpose(1, 2))
        return self.norm(self.fuse(torch.cat([seq1, seq2], dim=-1)))


def build_baseline_model(name, input_channels, num_poses, hidden_dim, num_layers, nhead, dropout, resnet_verson, num_keypoints=25, output_dim=2):
    name = name.lower()
    kwargs = dict(
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=nhead,
        num_poses=num_poses,
        num_keypoints=num_keypoints,
        output_dim=output_dim,
        dropout=dropout,
    )
    if name in ["pip_like_recon", "pip_like"]:
        return PIPLikeReconstructor(**kwargs)
    if name in ["tip_like_recon", "tip_like"]:
        return TIPLikeReconstructor(**kwargs)
    if name in ["imuposer_like_recon", "imuposer_like"]:
        return IMUPoserLikeReconstructor(**kwargs)
    if name in ["dynaip_like_recon", "dynaip_like"]:
        return DynaIPLikeReconstructor(**kwargs)
    if name in ["asip_like_recon", "asip_like"]:
        return ASIPLikeReconstructor(**kwargs)
    if name in ["mobileposer_like_recon", "mobileposer_like"]:
        return MobilePoserLikeReconstructor(**kwargs)
    raise ValueError(f"Unsupported reconstruction comparison model: {name}")
