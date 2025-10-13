import torch
from torch import nn
import math

class DETR(nn.Module):
    def __init__(self, num_queries, num_keypoints=25, dim_keypoint=2, input_channels=6, 
                 lstm_layers=2, hidden_dim=256, nheads=8, num_encoder_layers=3, num_decoder_layers=3,
                 max_seq_len=300):
        super().__init__()
        self.num_queries = num_queries
        self.num_keypoints = num_keypoints
        self.dim_keypoint = dim_keypoint
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # LSTM Backbone
        self.backbone = nn.LSTM(input_size=input_channels, hidden_size=hidden_dim//2, 
                                num_layers=lstm_layers, batch_first=True, bidirectional=True)

        # Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nheads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)

        # Learnable query embeddings
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # Linear output
        self.linear_keypoints = nn.Linear(hidden_dim, num_keypoints * dim_keypoint)

        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding', self._build_sinusoidal_position_encoding(max_seq_len, hidden_dim))

    def _build_sinusoidal_position_encoding(self, max_len, d_model):
        """生成正弦/余弦位置编码 [max_len, d_model]"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [max_len, d_model]

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]

        # LSTM backbone
        y, _ = self.backbone(x)  # [batch, seq_len, hidden_dim]
        y = y.permute(1, 0, 2)   # [seq_len, batch, hidden_dim]

        # Add sinusoidal position encoding
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")
        pos_encoding = self.pos_encoding[:seq_len, :].unsqueeze(1).repeat(1, batch_size, 1)
        y = y + pos_encoding

        # Decoder queries
        queries = self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch, hidden_dim]

        # Transformer
        out = self.transformer(y, queries)  # [num_queries, batch, hidden_dim]

        # Output
        out = out.permute(1, 0, 2)  # [batch, num_queries, hidden_dim]
        return self.linear_keypoints(out).view(batch_size, self.num_queries, self.num_keypoints, 2)


if __name__ == "__main__":
    batch_size = 2
    channels = 6
    time_len = 100
    num_queries = 3
    num_keypoints = 25

    model = DETR(num_queries, num_keypoints=25, dim_keypoint=2, input_channels=6, lstm_layers=2, hidden_dim=256, nheads=8, num_encoder_layers=3, num_decoder_layers=3)
    model.eval()

    inputs = torch.randn(batch_size, channels, time_len)
    keypoints = model(inputs)
    print("keypoints shape:", keypoints.shape)  # [batch, num_queries, num_keypoints, 2]