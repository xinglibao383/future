import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.metrics import PoseMetricTracker


def normalize(x, eps=1e-6):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True) + eps
    return (x - mean) / std


def evaluate_loss_mpjpe(model, dataloader, criterion, need_normalize):
    metric = Accumulator(2)
    pose_tracker = PoseMetricTracker()
    device = next(iter(model.parameters())).device
    model.eval()
    with torch.no_grad():
        for _, (x1, y1, z1, _, _, _) in enumerate(dataloader):
            batch_size = x1.shape[0]
            if need_normalize:
                x1 = normalize(x1)
            x1, y1, z1 = x1.to(device), y1.to(device), z1.to(device)
            y1_hat = model(x1)
            loss1 = criterion(y1_hat, y1)
            metric.add(loss1.item() * batch_size, batch_size)
            pose_tracker.update(y1_hat, y1, z1)
    return {
        "loss": metric[0] / metric[1],
        **pose_tracker.summary(),
    }


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()
    min_val_mpjpe, best_epoch = float('inf'), 0
    best_val_metrics = None
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        train_pose_tracker = PoseMetricTracker()
        model.train()
        for i, (x1, y1, z1, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x1.shape[0]
            if need_normalize:
                x1 = normalize(x1)
            mask = torch.rand_like(x1) >= mask_ratio
            x1 = x1 * mask.float()
            x1, y1, z1 = x1.to(devices[0]), y1.to(devices[0]), z1.to(devices[0])
            y1_hat = model(x1)
            loss1 = criterion(y1_hat, y1)
            loss1.backward()
            optimizer.step()
            metric.add(loss1.item() * batch_size, batch_size)
            train_pose_tracker.update(y1_hat.detach(), y1, z1)
            if i != 0 and i % 20 == 0:
                train_pose_metrics = train_pose_tracker.summary()
                train_loss1 = metric[0] / metric[1]
                print(
                    f'Epoch: {epoch}, iter: {i}, train loss: {train_loss1:.4f}, '
                    f'train mpjpe: {train_pose_metrics["mpjpe"]:.4f}, '
                    f'{train_pose_tracker.format_pck_metrics(train_pose_metrics)}'
                )
        train_pose_metrics = train_pose_tracker.summary()
        train_loss1 = metric[0] / metric[1]
        val_metrics = evaluate_loss_mpjpe(model, val_loader, criterion, need_normalize)
        train_msg = (
            f'[{timestamp}] Epoch: {epoch}, train loss: {train_loss1:.4f}, '
            f'train mpjpe: {train_pose_metrics["mpjpe"]:.4f}, '
            f'{train_pose_tracker.format_pck_metrics(train_pose_metrics)}'
        )
        val_msg = (
            f'[{timestamp}] Epoch: {epoch},   val loss: {val_metrics["loss"]:.4f},   '
            f'val mpjpe: {val_metrics["mpjpe"]:.4f}, '
            f'{train_pose_tracker.format_pck_metrics(val_metrics)}'
        )
        train_joint_msg = (
            f'[{timestamp}] Epoch: {epoch}, train per-joint MPJPE:\n'
            f'{train_pose_tracker.format_per_joint_mpjpe(train_pose_metrics["per_joint_mpjpe"])}'
        )
        val_joint_msg = (
            f'[{timestamp}] Epoch: {epoch},   val per-joint MPJPE:\n'
            f'{train_pose_tracker.format_per_joint_mpjpe(val_metrics["per_joint_mpjpe"])}'
        )
        print(train_msg)
        print(val_msg)
        print(train_joint_msg)
        print(val_joint_msg)
        logger.record([train_msg])
        logger.record([val_msg])
        logger.record([train_joint_msg], print_flag=False)
        logger.record([val_joint_msg], print_flag=False)
        if val_metrics["mpjpe"] < min_val_mpjpe:
            min_val_mpjpe = val_metrics["mpjpe"]
            best_epoch = epoch
            best_val_metrics = val_metrics
        if epoch - best_epoch >= 20:
            break
    logger.record([f'[{timestamp}] The best mpjpe occurred in epoch {best_epoch}'])
    if best_val_metrics is not None:
        best_msg = (
            f'[{timestamp}] Best val pose metrics: mpjpe={best_val_metrics["mpjpe"]:.4f}, '
            f'{train_pose_tracker.format_pck_metrics(best_val_metrics)}'
        )
        logger.record([best_msg])
        logger.record([
            f'[{timestamp}] Best val per-joint MPJPE:\n'
            f'{train_pose_tracker.format_per_joint_mpjpe(best_val_metrics["per_joint_mpjpe"])}'
        ], print_flag=False)
    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{best_epoch}.pth")