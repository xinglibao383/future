import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, d_state, d_conv, expand, target_len, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_len = target_len
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_norm = nn.LayerNorm(input_dim)
        self.output_head = nn.Linear(input_dim, output_dim)
        self.feedback_proj = nn.Linear(input_dim, input_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x):
        h = self.input_norm(self.input_proj(x) + self.start_token)
        outputs = []
        for _ in range(self.target_len):
            m_out = self.mamba(h)   # (B, 1, D)
            gate = self.residual_gate(h)
            h = gate * h + (1.0 - gate) * m_out
            h = self.hidden_norm(h)
            h = self.dropout(h)
            y_t = self.output_head(h)   # (B, 1, output_dim)
            outputs.append(y_t)
            h = self.feedback_proj(h)
        y = torch.cat(outputs, dim=1)   # (B, target_len, output_dim)
        y = y.permute(0, 2, 1)          # (B, output_dim, target_len)
        return y


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/models/generator/mamba.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, 1, 512).to(device)
    model = MambaGenerator(input_dim=512, output_dim=30, d_state=64, d_conv=4, expand=2, target_len=15, dropout=0.1).to(device)
    y = model(x)
    print(y.shape)