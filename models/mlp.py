import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, feature_dim, hidden_size, output_len, output_dim):
        super(MLP, self).__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_len * output_dim)
        )

    def forward(self, x):
        out = self.decoder(x)
        out = out.view(-1, self.output_len, self.output_dim)
        return out
