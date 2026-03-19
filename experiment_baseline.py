import datetime
import torch
from utils.logger import Logger
from utils.dataloader import *
from utils.train_baseline import train
from models.comparison.baselines import build_baseline_model


torch.manual_seed(3407)
# devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
devices = [torch.device('cuda:3')]
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputs/experiment/baseline'
data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'


def experiment_baseline(baseline="aipose"):
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


# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_baseline.py
if __name__ == "__main__":
    # experiment_baseline(baseline="aipose")
    # experiment_baseline(baseline="pip_like_recon")
    # experiment_baseline(baseline="tip_like_recon")
    experiment_baseline(baseline="imuposer_like_recon")
    experiment_baseline(baseline="dynaip_like_recon")
    experiment_baseline(baseline="asip_like_recon")
    experiment_baseline(baseline="mobileposer_like_recon")