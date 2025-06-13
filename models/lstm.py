import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, len_input, len_predict, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_joints = 25
        self.len_input = len_input
        self.len_predict = len_predict
        self.input_size = self.num_joints * 2
        self.output_size = self.num_joints * 2
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        y, _ = self.lstm(x.view(batch_size, self.len_input, -1))
        y = self.fc(y[:, -1, :].unsqueeze(1).repeat(1, self.len_predict, 1))
        return y.view(batch_size, self.len_predict, self.num_joints, 2)
    
if __name__ == "__main__":
    model = LSTM(len_input=10, len_predict=10, hidden_size=128, num_layers=2)
    x = torch.randn(32, 60, 25, 3)
    x = x[:, :, :, :2]
    y = model(x)
    print(y.shape)