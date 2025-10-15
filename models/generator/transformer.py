import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=30, num_layers=4, nhead=8, dropout=0.1, target_len=50):
        super().__init__()
        self.target_len = target_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1)]
        memory = self.encoder(x)
        queries = self.pos_encoding[:, :self.target_len].expand(x.size(0), self.target_len, -1)
        out = self.encoder(queries + memory.mean(dim=1, keepdim=True))
        return self.output_proj(out).permute(0, 2, 1)


if __name__ == "__main__":
    x = torch.randn(32, 1, 512)
    model = TransformerGenerator(input_dim=512, hidden_dim=256, output_dim=30, num_layers=4, nhead=8, dropout=0.1, target_len=50)
    y = model(x)
    print(y.shape)