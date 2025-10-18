import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaGenerator(nn.Module):
    def __init__(self, input_dim, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x.permute(0, 2, 1)).permute(0, 2, 1)
        

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/classify/models/mamba.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, 30, 45).to(device)
    model = MambaGenerator(input_dim=30, d_state=64, d_conv=4, expand=2).to(device)
    y = model(x)
    print(y.shape)