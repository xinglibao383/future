import torch
import torch.nn as nn


class MLPC(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_classes, dropout=0.3):
        super(MLPC, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.decoder(x)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # 用于将 hidden 映射回原始特征维度
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        x: 输入序列 (batch_size, input_len, input_dim)
        输出: 预测序列 (batch_size, pred_len, input_dim)
        """
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)

        # Encoder: 提取历史信息
        lstm_out, (h, c) = self.lstm(x)

        # 使用最后时间步的输出（也可以用 mean pooling）
        global_feat = lstm_out[:, -1, :]

        # Decoder: 初始输入为零向量
        decoder_input = torch.zeros((batch_size, 1, self.input_dim), device=x.device)

        outputs = []
        for _ in range(self.pred_len):
            out, (h, c) = self.lstm(decoder_input, (h, c))  # out: (batch, 1, hidden_dim)
            pred = self.out_proj(out)  # (batch, 1, input_dim)
            outputs.append(pred)
            decoder_input = pred

        return global_feat, torch.cat(outputs, dim=1).permute(0, 2, 1)  # (batch, pred_len, input_dim)
    

class IMUPose(nn.Module):
    def __init__(self, hidden_dim, num_layers, len_output, dropout):
        super(IMUPose, self).__init__()
        self.imu_channel = 5 * 6
        self.lstm = LSTM(input_dim=self.imu_channel, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=int(len_output*50))
        self.imu_classfier = MLPC(hidden_dim, hidden_dim * 2, 34, dropout)

    def forward(self, imu):
        imu_feature, imu2 = self.lstm(imu)
        label1 = self.imu_classfier(imu_feature)
        # todo 可以把之前的imu拼一些给imu2
        imu2_feature, _ = self.lstm(imu2)
        label2 = self.imu_classfier(imu2_feature)
        return label1, imu2, label2

if __name__ == "__main__":
    len_input = 1.5
    x = torch.randn(32, 30, int(len_input*50))
    model = IMUPose(len_input)
    label1, imu2, label2 = model(x)
    print(label1.shape, imu2.shape, label2.shape)