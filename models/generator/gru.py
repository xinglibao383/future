import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, target_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = []
        h = torch.zeros(self.gru.num_layers, x.size(0), self.hidden_dim, device=x.device)
        input_t = x
        for _ in range(self.target_len):
            out, h = self.gru(input_t, h)
            outputs.append(self.fc(out))
            input_t = outputs[-1]
        outputs = torch.cat(outputs, dim=1)
        return self.proj(outputs).permute(0, 2, 1)


if __name__ == "__main__":
    x = torch.randn(32, 1, 512)
    model = GRUGenerator(input_dim=512, hidden_dim=256, output_dim=30, num_layers=2, dropout=0.2, target_len=50)
    y = model(x)
    print(y.shape)