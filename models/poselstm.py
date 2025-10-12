import torch
from torch import nn

class PoseLSTM(nn.Module):
    def __init__(self, input_size=30, hidden_size=256, num_layers=2, num_keypoints=25, output_size=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keypoints = num_keypoints
        self.output_size = output_size

        # LSTM: 输入 [B, T, C]，所以需要转置输入
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)

        # 一个简单的线性层，将 LSTM 输出映射到关键点
        # 我们取最后时间步的 hidden state 映射到 [num_keypoints * output_size]
        self.fc = nn.Linear(hidden_size, num_keypoints * output_size)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        # LSTM forward
        out, (h_n, c_n) = self.lstm(x)  # out: [B, T, hidden_size]

        # 取最后一个时间步的输出
        last_out = out[:, -1, :]  # [B, hidden_size]

        # 全连接映射到关键点
        keypoints = self.fc(last_out)  # [B, num_keypoints*output_size]
        keypoints = keypoints.view(-1, self.num_keypoints, self.output_size)  # [B, num_keypoints, 2]

        return keypoints


# ===== 测试 =====
if __name__ == "__main__":
    model = PoseLSTM(input_size=30, hidden_size=256, num_layers=2, num_keypoints=25, output_size=2)
    model.eval()

    inputs = torch.randn(32, 30, 800)  # batch_size=32, channels=30, time=800
    with torch.no_grad():
        keypoints = model(inputs)

    print("输出形状:", keypoints.shape)  # [32, 25, 2]