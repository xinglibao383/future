import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.metrics24 import PoseMetricTracker


def normalize(x, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std


def evaluate_loss_mjpe(model, dataloader, criterion, need_normalize):
    metric = Accumulator(2)
    pose_tracker = PoseMetricTracker(enable_ratio_pck=False, enable_pixel_pck=False)
    device = next(iter(model.parameters())).device
    model.eval()

    with torch.no_grad():
        for _, (x1, y1, _, _) in enumerate(dataloader):
            batch_size = x1.shape[0]

            x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
            y1 = torch.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)

            if need_normalize:
                x1 = normalize(x1)

            x1 = x1.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)

            y1_hat = model(x1)
            y1_hat = torch.nan_to_num(y1_hat, nan=0.0, posinf=0.0, neginf=0.0)

            loss1 = criterion(y1_hat, y1)
            metric.add(loss1.item() * batch_size, batch_size)
            pose_tracker.update(y1_hat, y1, None)

    return {"loss": metric[0] / metric[1], **pose_tracker.summary()}


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize,
          num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()

    min_val_mjpe, best_epoch = float("inf"), 0
    best_val_metrics = None

    for epoch in range(num_epochs):
        metric = Accumulator(2)
        train_pose_tracker = PoseMetricTracker(enable_ratio_pck=False, enable_pixel_pck=False)
        model.train()

        for i, (x1, y1, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x1.shape[0]

            x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
            y1 = torch.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)

            if need_normalize:
                x1 = normalize(x1)

            mask = (torch.rand_like(x1) >= mask_ratio).float()
            x1 = x1 * mask

            x1 = x1.to(devices[0], non_blocking=True)
            y1 = y1.to(devices[0], non_blocking=True)

            y1_hat = model(x1)
            y1_hat = torch.nan_to_num(y1_hat, nan=0.0, posinf=0.0, neginf=0.0)

            loss1 = criterion(y1_hat, y1)
            loss1.backward()
            optimizer.step()

            metric.add(loss1.item() * batch_size, batch_size)
            train_pose_tracker.update(y1_hat.detach(), y1, None)

            if i != 0 and i % 20 == 0:
                train_pose_metrics = train_pose_tracker.summary()
                train_loss1 = metric[0] / metric[1]
                lines = train_pose_tracker.format_pose_metric_lines(
                    train_pose_metrics,
                    mjpe_label="train mjpe",
                    pixel_pck_label="train pixel pck",
                )
                print(f"Epoch: {epoch}, iter: {i}, train loss: {train_loss1:.4f}, {lines[0]}")
                if len(lines) > 1:
                    print(f"Epoch: {epoch}, iter: {i}, {lines[1]}")

        train_pose_metrics = train_pose_tracker.summary()
        train_loss1 = metric[0] / metric[1]
        val_metrics = evaluate_loss_mjpe(model, val_loader, criterion, need_normalize)

        train_lines = train_pose_tracker.format_pose_metric_lines(
            train_pose_metrics,
            mjpe_label="train mjpe",
            pixel_pck_label="train pixel pck",
        )
        val_lines = train_pose_tracker.format_pose_metric_lines(
            val_metrics,
            mjpe_label="val mjpe",
            pixel_pck_label="val pixel pck",
        )

        train_msg_1 = f"[{timestamp}] Epoch: {epoch}, train loss: {train_loss1:.4f}, {train_lines[0]}"
        val_msg_1 = f"[{timestamp}] Epoch: {epoch}, val loss: {val_metrics['loss']:.4f}, {val_lines[0]}"

        is_best_epoch = val_metrics["mjpe"] < min_val_mjpe
        if is_best_epoch:
            min_val_mjpe = val_metrics["mjpe"]
            best_epoch = epoch
            best_val_metrics = val_metrics

        print(train_msg_1)
        print(val_msg_1)
        logger.record([train_msg_1])
        logger.record([val_msg_1])

        if is_best_epoch:
            if len(train_lines) > 1:
                train_msg_2 = f"[{timestamp}] Epoch: {epoch}, {train_lines[1]}"
                print(train_msg_2)
                logger.record([train_msg_2])

            if len(val_lines) > 1:
                val_msg_2 = f"[{timestamp}] Epoch: {epoch}, {val_lines[1]}"
                print(val_msg_2)
                logger.record([val_msg_2])

            train_joint_msg = (
                f"[{timestamp}] Epoch: {epoch}, train per-joint MJPE:\n"
                f"{train_pose_tracker.format_per_joint_mjpe(train_pose_metrics['per_joint_mjpe'])}"
            )
            val_joint_msg = (
                f"[{timestamp}] Epoch: {epoch}, val per-joint MJPE:\n"
                f"{train_pose_tracker.format_per_joint_mjpe(val_metrics['per_joint_mjpe'])}"
            )

            print(train_joint_msg)
            print(val_joint_msg)

            logger.record([train_joint_msg], print_flag=False)
            logger.record([val_joint_msg], print_flag=False)

            ckpt_dir = os.path.join(output_save_path, timestamp, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        if epoch - best_epoch >= 20:
            break

    logger.record([f"[{timestamp}] The best mjpe occurred in epoch {best_epoch}"])

    if best_val_metrics is not None:
        best_lines = train_pose_tracker.format_pose_metric_lines(
            best_val_metrics,
            mjpe_label="best val mjpe",
            pixel_pck_label="best val pixel pck",
        )
        logger.record([f"[{timestamp}] {best_lines[0]}"])
        if len(best_lines) > 1:
            logger.record([f"[{timestamp}] {best_lines[1]}"])
        logger.record([
            f"[{timestamp}] Best val per-joint MJPE:\n"
            f"{train_pose_tracker.format_per_joint_mjpe(best_val_metrics['per_joint_mjpe'])}"
        ], print_flag=False)

    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{best_epoch}.pth")