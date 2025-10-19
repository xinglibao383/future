import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, d_state, d_conv, expand):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim1), nn.GELU(), nn.Linear(hidden_dim1, hidden_dim2))
        self.mamba = Mamba(d_model=hidden_dim2, d_state=d_state, d_conv=d_conv, expand=expand)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim2, hidden_dim1), nn.GELU(), nn.Linear(hidden_dim1, input_dim))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.mamba(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
        

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/classify/models/mamba.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, 30, 45).to(device)
    model = MambaGenerator(input_dim=30, hidden_dim1=64, hidden_dim2=128, output_dim=30, d_state=64, d_conv=4, expand=2).to(device)
    y = model(x)
    print(y.shape)