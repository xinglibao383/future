import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, len_predict, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_joints = 25
        self.len_predict = len_predict
        self.lstm = nn.LSTM(
            input_size=self.num_joints * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, self.num_joints * 2)

    def forward(self, x):
        batch_size, input_len = x.shape[0], x.shape[1]
        y, _ = self.lstm(x.reshape(batch_size, input_len, self.num_joints * 2))
        y = self.fc(y[:, -1, :].unsqueeze(1).repeat(1, self.len_predict, 1))
        return y.view(batch_size, -1, self.num_joints, 2)
    
if __name__ == "__main__":
    model = LSTM(len_predict=15, hidden_size=512, num_layers=3)
    x = torch.randn(32, 60, 25, 3)
    x = x[:, :, :, :2]
    y = model(x)
    print(y.shape)