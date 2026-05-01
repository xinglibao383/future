import os
import csv
import datetime
import torch
import shutil
from utils.logger import Logger
from utils.dataloader import *
from utils.train3 import train as train3
from models.posenet import *
from utils.predict_max_predict_len import *
from draw2028.draw_max_predict_len import *


def append_csv(csv_path, noise_steps, noise_std, noise_type, data):
    ns = "None" if noise_steps is None else "_".join(map(str, noise_steps))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ns, noise_std, noise_type, *data])


def accumulate_error(checkpoint_filepath=None, noise_steps=None, noise_std=0.0, noise_type="gaussian"):
    devices = [torch.device('cuda:0'), torch.device('cuda:2'), torch.device('cuda:1'), torch.device('cuda:3')]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = '/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/accumulate_error'
    data_root_path = '/mnt/mydata/yh/liming/workspace/future/mydata'
    logger = Logger(save_path=output_save_path, timestamp=timestamp)
    logger.record([f'备注: 累积误差实验, noise_steps={noise_steps}, noise_std={noise_std}, noise_type={noise_type}'])
    mask_ratio, batch_size, lr, num_epochs, loss_func = 0.25, 256, 1e-3, 200, "l1"
    resnet_verson, imu_generator = "resnet18", "transformer"
    transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout = 128, 2, 4, 0.1
    use_len, compute_len, predict_len, stride_len = 60, 15, 15, 15
    need_normalize, alpha, beta, gamma = True, 1, 1, 1
    params = {
        "checkpoint_filepath": checkpoint_filepath,
        "mask_ratio": mask_ratio, "batch_size": batch_size, "lr": lr, "epochs": num_epochs, "loss_func": loss_func,
        "resnet_verson": resnet_verson, "imu_generator": imu_generator, 
        "transformer_hidden": transformer_hidden, "transformer_layers": transformer_layers, "transformer_nhead": transformer_nhead, "transformer_dropout": transformer_dropout, 
        "use_len": use_len, "compute_len": compute_len, "predict_len": predict_len, "stride_len": stride_len,
        "need_normalize": need_normalize, "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    logger.record([", ".join([f"{k}={v}" for k, v in params.items()])])
    imu_generator_params = (transformer_hidden, transformer_layers, transformer_nhead, transformer_dropout)
    model = PoseNet(input_channels=30, resnet_verson=resnet_verson, imu_generator=imu_generator, imu_generator_params=imu_generator_params, target_time=50, target_poses=15, num_poses=compute_len, num_keypoints=25, output_dim=2)
    if checkpoint_filepath is not None:
        _, val_loader = get_dataloaders_v3(data_root_path, use_len, compute_len, 150, stride_len, batch_size, 0.8, random_seed=3407)
        data = val_loss_mpjpe(model, checkpoint_filepath, devices[0], val_loader, logger, noise_steps, noise_std, noise_type)
        data = [round(x, 2) for x in data]
        shutil.rmtree(os.path.join(output_save_path, f"{timestamp}"))
        plot_prediction_horizon(save_path=os.path.join(output_save_path, f"{noise_steps}_{noise_std}_{noise_type}.png"), y=data)
        append_csv("/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/accumulate_error/data.csv", noise_steps, noise_std, noise_type, data)
    else:
        train_loader, val_loader = get_dataloaders_v3_max_predict_len(data_root_path, use_len, compute_len, 150, stride_len, batch_size, 0.8)
        train3(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp)


# nohup /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_accumulate_error.py > /dev/null 2>&1 &
# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/experiment_accumulate_error.py
if __name__ == "__main__":
    for noise_type in ["gaussian", "uniform", "scale"]:
        for noise_std in range(10):
            accumulate_error(
                checkpoint_filepath="/mnt/mydata/yh/liming/workspace/future/outputs/experiment2028/accumulate_error/20260501003207/checkpoints/epoch_197.pth",
                noise_steps=[4], 
                noise_std=noise_std, 
                noise_type=noise_type
            )