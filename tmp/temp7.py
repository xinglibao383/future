import torch
from models.posenet import *

if __name__ == "__main__":
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 128, 1e-2, 800, "l1"
    resnet_verson, lstm_hidden, lstm_layers, lstm_dropout = "resnet18", 128, 2, 0
    use_len, compute_len, predict_len, stride_len = 45, 15, 15, 15
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    
    x = torch.randn(32, 30, 150)
    now_pose, future_x, future_pose = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", now_pose.shape)
    print("输出形状:", future_x.shape)
    print("输出形状:", future_pose.shape)