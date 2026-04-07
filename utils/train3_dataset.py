import os
import torch
import shutil
import matplotlib.pyplot as plt
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.metrics import PoseMetricTracker


def normalize(x, eps=1e-6):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True) + eps
    return (x - mean) / std


def evaluate_loss_mpjpe(model, dataloader, criterion, need_normalize, timestamp, output_save_path):
    metric = Accumulator(4)
    pose_tracker = PoseMetricTracker(prefixes=["current", "future"], enable_ratio_pck=False, enable_pixel_pck=True)
    device = next(iter(model.parameters())).device
    model.eval()
    with torch.no_grad():
        for i, (x1, y1, x2, y2) in enumerate(dataloader):
            batch_size = x1.shape[0]
            if need_normalize:
                x1, x2 = normalize(x1), normalize(x2)
            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)
            y1_hat, x2_hat, y2_hat = model(x1)
            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(x2_hat, x2)
            loss3 = criterion(y2_hat, y2)
            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size)
            pose_tracker.update(y1_hat, y1, None, prefix="current")
            pose_tracker.update(y2_hat, y2, None, prefix="future")

    return {
        "loss1": metric[0] / metric[3],
        "loss2": metric[1] / metric[3],
        "loss3": metric[2] / metric[3],
        **pose_tracker.summary(),
    }


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp):
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
        metric = Accumulator(4)
        train_pose_tracker = PoseMetricTracker(prefixes=["current", "future"], enable_ratio_pck=False, enable_pixel_pck=True)
        model.train()
        for i, (x1, y1, x2, y2) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x1.shape[0]
            if need_normalize:
                x1, x2 = normalize(x1), normalize(x2)
            mask = torch.rand_like(x1) >= mask_ratio
            x1 = x1 * mask.float()
            x1, y1 = x1.to(devices[0]), y1.to(devices[0])
            x2, y2 = x2.to(devices[0]), y2.to(devices[0])
            y1_hat, x2_hat, y2_hat = model(x1)

            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(x2_hat, x2)
            loss3 = criterion(y2_hat, y2)
            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            loss.backward()
            optimizer.step()

            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size)
            train_pose_tracker.update(y1_hat.detach(), y1, None, prefix="current")
            train_pose_tracker.update(y2_hat.detach(), y2, None, prefix="future")

            if i != 0 and i % 20 == 0:
                train_pose_metrics = train_pose_tracker.summary()
                train_loss1 = metric[0] / metric[3]
                train_loss2 = metric[1] / metric[3]
                train_loss3 = metric[2] / metric[3]
                current_lines = train_pose_tracker.format_pose_metric_lines(
                    train_pose_metrics,
                    prefix="current",
                    mpjpe_label="train current mpjpe",
                    pixel_pck_label="train current pixel pck",
                )
                future_lines = train_pose_tracker.format_pose_metric_lines(
                    train_pose_metrics,
                    prefix="future",
                    mpjpe_label="train future mpjpe",
                    pixel_pck_label="train future pixel pck",
                )
                print(
                    f'Epoch: {epoch}, iter: {i}, '
                    f'train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, '
                    f'{current_lines[0]}, {future_lines[0]}'
                )
                # Only show pixel PCK during iteration logging when this branch is re-enabled later.
                # if len(current_lines) > 1 and len(future_lines) > 1:
                #     print(f'Epoch: {epoch}, iter: {i}, {current_lines[1]}, {future_lines[1]}')

        train_pose_metrics = train_pose_tracker.summary()
        train_loss1 = metric[0] / metric[3]
        train_loss2 = metric[1] / metric[3]
        train_loss3 = metric[2] / metric[3]
        val_metrics = evaluate_loss_mpjpe(model, val_loader, criterion, need_normalize, timestamp, output_save_path)

        train_current_lines = train_pose_tracker.format_pose_metric_lines(
            train_pose_metrics,
            prefix="current",
            mpjpe_label="train current mpjpe",
            pixel_pck_label="train current pixel pck",
        )
        train_future_lines = train_pose_tracker.format_pose_metric_lines(
            train_pose_metrics,
            prefix="future",
            mpjpe_label="train future mpjpe",
            pixel_pck_label="train future pixel pck",
        )
        val_current_lines = train_pose_tracker.format_pose_metric_lines(
            val_metrics,
            prefix="current",
            mpjpe_label="val current mpjpe",
            pixel_pck_label="val current pixel pck",
        )
        val_future_lines = train_pose_tracker.format_pose_metric_lines(
            val_metrics,
            prefix="future",
            mpjpe_label="val future mpjpe",
            pixel_pck_label="val future pixel pck",
        )

        train_msg_1 = (
            f'[{timestamp}] Epoch: {epoch}, '
            f'train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, '
            f'{train_current_lines[0]}, {train_future_lines[0]}'
        )
        val_msg_1 = (
            f'[{timestamp}] Epoch: {epoch}, '
            f'val loss1: {val_metrics["loss1"]:.4f}, val loss2: {val_metrics["loss2"]:.4f}, val loss3: {val_metrics["loss3"]:.4f}, '
            f'{val_current_lines[0]}, {val_future_lines[0]}'
        )

        current_score = val_metrics["current_mpjpe"] + val_metrics["future_mpjpe"]
        is_best_epoch = current_score < min_val_mpjpe
        if is_best_epoch:
            min_val_mpjpe = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics

        print(train_msg_1)
        print(val_msg_1)
        logger.record([train_msg_1])
        logger.record([val_msg_1])

        if is_best_epoch:
            train_msg_2 = f'[{timestamp}] Epoch: {epoch}, {train_current_lines[1]}, {train_future_lines[1]}'
            val_msg_2 = f'[{timestamp}] Epoch: {epoch}, {val_current_lines[1]}, {val_future_lines[1]}'

            train_current_joint_msg = (
                f'[{timestamp}] Epoch: {epoch}, train current per-joint MPJPE:\n'
                f'{train_pose_tracker.format_per_joint_mpjpe(train_pose_metrics["current_per_joint_mpjpe"])}'
            )
            train_future_joint_msg = (
                f'[{timestamp}] Epoch: {epoch}, train future per-joint MPJPE:\n'
                f'{train_pose_tracker.format_per_joint_mpjpe(train_pose_metrics["future_per_joint_mpjpe"])}'
            )
            val_current_joint_msg = (
                f'[{timestamp}] Epoch: {epoch}, val current per-joint MPJPE:\n'
                f'{train_pose_tracker.format_per_joint_mpjpe(val_metrics["current_per_joint_mpjpe"])}'
            )
            val_future_joint_msg = (
                f'[{timestamp}] Epoch: {epoch}, val future per-joint MPJPE:\n'
                f'{train_pose_tracker.format_per_joint_mpjpe(val_metrics["future_per_joint_mpjpe"])}'
            )

            print(train_msg_2)
            print(val_msg_2)
            print(train_current_joint_msg)
            print(train_future_joint_msg)
            print(val_current_joint_msg)
            print(val_future_joint_msg)

            logger.record([train_msg_2])
            logger.record([val_msg_2])
            logger.record([train_current_joint_msg], print_flag=False)
            logger.record([train_future_joint_msg], print_flag=False)
            logger.record([val_current_joint_msg], print_flag=False)
            logger.record([val_future_joint_msg], print_flag=False)

        if epoch - best_epoch >= 20:
            break

    logger.record([f'[{timestamp}] The best mpjpe occurred in epoch {best_epoch}'])
    if best_val_metrics is not None:
        best_current_lines = train_pose_tracker.format_pose_metric_lines(
            best_val_metrics,
            prefix="current",
            mpjpe_label="best val current mpjpe",
            pixel_pck_label="best val current pixel pck",
        )
        best_future_lines = train_pose_tracker.format_pose_metric_lines(
            best_val_metrics,
            prefix="future",
            mpjpe_label="best val future mpjpe",
            pixel_pck_label="best val future pixel pck",
        )
        logger.record([f'[{timestamp}] {best_current_lines[0]}, {best_future_lines[0]}'])
        if len(best_current_lines) > 1 and len(best_future_lines) > 1:
            logger.record([f'[{timestamp}] {best_current_lines[1]}, {best_future_lines[1]}'])
        logger.record([
            f'[{timestamp}] Best val current per-joint MPJPE:\n'
            f'{train_pose_tracker.format_per_joint_mpjpe(best_val_metrics["current_per_joint_mpjpe"])}'
        ], print_flag=False)
        logger.record([
            f'[{timestamp}] Best val future per-joint MPJPE:\n'
            f'{train_pose_tracker.format_per_joint_mpjpe(best_val_metrics["future_per_joint_mpjpe"])}'
        ], print_flag=False)
    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{best_epoch}.pth")
