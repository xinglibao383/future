import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train_baseline import train
from models.comparison.baselines import build_baseline_model
from utils.train3 import train as train3
from models.posenet import *


torch.manual_seed(3407)
devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline'
data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'


def ours():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 基线模型对比实验, baseline=aipose'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=int(predict_len / 15 * 50), target_poses=predict_len, num_poses=compute_len, num_keypoints=25, output_dim=2)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, 0.8, random_seed=3407)
    train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


def experiment_baseline(baseline="aipose"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 基线模型对比实验, baseline={baseline}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, hidden_dim, num_layers, nhead, dropout = "resnet18", 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    input_channels, train_ratio, need_normalize = 30, 0.8, True
    params = {
        "baseline": baseline, "resnet_verson": resnet_verson,
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "nhead": nhead, "dropout": dropout,
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "input_channels": input_channels, "train_ratio": train_ratio, 
        "need_normalize": need_normalize,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    model = build_baseline_model(name=baseline, input_channels=input_channels, num_poses=compute_len, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout, resnet_verson=resnet_verson)
    train_loader, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, predict_len, stride_len, batch_size, train_ratio, random_seed=3407)
    train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, num_epochs, devices, output_save_path, logger, timestamp)


# nohup /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_baseline.py > /dev/null 2>&1 &
# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_baseline.py
if __name__ == "__main__":
    ours()
    # experiment_baseline(baseline="aipose")
    # experiment_baseline(baseline="pip_like_recon")
    # experiment_baseline(baseline="tip_like_recon")
    # experiment_baseline(baseline="dynaip_like_recon")
    # experiment_baseline(baseline="asip_like_recon")
    # experiment_baseline(baseline="mobileposer_like_recon")
    # experiment_baseline(baseline="imuposer_like_recon")