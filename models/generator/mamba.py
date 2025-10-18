import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, d_state, d_conv, expand, target_len):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_len = target_len

        self.mamba = Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        device = x.device
        input_t = x.permute(0, 2, 1)    # batch_size * time_len * channels
        outputs = []
        for _ in range(self.target_len):
            out = self.mamba(input_t)
            outputs.append(out[:, -1:, :])
            input_t = torch.cat([input_t[:, 1:, :], outputs[-1]], dim=1)
        return torch.cat(outputs, dim=1).permute(0, 2, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 30, 60).to(device)
    model = MambaGenerator(input_dim=30, output_dim=30, d_state=64, d_conv=4, expand=2, target_len=15).to(device)
    y = model(x)
    print(y.shape)