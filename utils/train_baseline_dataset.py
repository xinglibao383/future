import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.metrics_dip import DIPMetricTracker

try:
    from utils.smpl_eval import SMPLForward
except Exception:
    SMPLForward = None


SMPL_MODEL_PATH = "/mnt/mydata/yh/liming/workspace/future/SMPL"
SMPL_GENDER = "neutral"


def normalize(x, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std


def build_smpl_forward(device):
    if SMPL_MODEL_PATH is None:
        return None
    if SMPLForward is None:
        return None
    try:
        return SMPLForward(
            model_path=SMPL_MODEL_PATH,
            gender=SMPL_GENDER,
            device=str(device),
        )
    except Exception as exc:
        print(f"[Warning] Failed to initialize SMPLForward: {exc}")
        return None


def evaluate_loss_dip(model, dataloader, criterion, need_normalize, smpl_forward=None):
    metric = Accumulator(2)
    dip_tracker = DIPMetricTracker()
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

            loss = criterion(y1_hat, y1)
            metric.add(loss.item() * batch_size, batch_size)

            dip_tracker.update_angle(y1_hat, y1)

            if smpl_forward is not None:
                pred_joints = smpl_forward.pose_to_joints(y1_hat)[..., :24, :]
                gt_joints = smpl_forward.pose_to_joints(y1)[..., :24, :]
                dip_tracker.update_pos(pred_joints, gt_joints)

    return {"loss": metric[0] / metric[1], **dip_tracker.summary()}


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize,
          num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()

    smpl_forward = build_smpl_forward(devices[0])

    min_val_ang_err = float("inf")
    best_epoch = 0
    best_val_metrics = None

    for epoch in range(num_epochs):
        metric = Accumulator(2)
        train_tracker = DIPMetricTracker()
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

            loss = criterion(y1_hat, y1)
            loss.backward()
            optimizer.step()

            metric.add(loss.item() * batch_size, batch_size)

            train_tracker.update_angle(y1_hat.detach(), y1)

            if smpl_forward is not None:
                pred_joints = smpl_forward.pose_to_joints(y1_hat.detach())[..., :24, :]
                gt_joints = smpl_forward.pose_to_joints(y1)[..., :24, :]
                train_tracker.update_pos(pred_joints, gt_joints)

            if i != 0 and i % 20 == 0:
                train_metrics = train_tracker.summary()
                train_loss = metric[0] / metric[1]

                angle_line = train_tracker.format_angle_metrics(
                    train_metrics,
                    label="train mean joint angle error"
                )
                pos_line = train_tracker.format_pos_metrics(
                    train_metrics,
                    label="train positional error"
                )

                print(
                    f"Epoch: {epoch}, iter: {i}, "
                    f"train loss: {train_loss:.4f}, {angle_line}, {pos_line}"
                )

        train_metrics = train_tracker.summary()
        train_loss = metric[0] / metric[1]
        val_metrics = evaluate_loss_dip(model, val_loader, criterion, need_normalize, smpl_forward)

        train_angle_line = train_tracker.format_angle_metrics(
            train_metrics,
            label="train mean joint angle error"
        )
        train_pos_line = train_tracker.format_pos_metrics(
            train_metrics,
            label="train positional error"
        )

        val_angle_line = train_tracker.format_angle_metrics(
            val_metrics,
            label="val mean joint angle error"
        )
        val_pos_line = train_tracker.format_pos_metrics(
            val_metrics,
            label="val positional error"
        )

        train_msg = (
            f"[{timestamp}] Epoch: {epoch}, "
            f"train loss: {train_loss:.4f}, "
            f"{train_angle_line}, {train_pos_line}"
        )
        val_msg = (
            f"[{timestamp}] Epoch: {epoch}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"{val_angle_line}, {val_pos_line}"
        )

        is_best_epoch = val_metrics["ang_err_deg"] < min_val_ang_err
        if is_best_epoch:
            min_val_ang_err = val_metrics["ang_err_deg"]
            best_epoch = epoch
            best_val_metrics = val_metrics

        print(train_msg)
        print(val_msg)
        logger.record([train_msg])
        logger.record([val_msg])

        if is_best_epoch:
            train_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, train per-joint angle error (deg):\n"
                f"{train_tracker.format_per_joint_angle(train_metrics['per_joint_ang_err_deg'])}"
            )
            val_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, val per-joint angle error (deg):\n"
                f"{train_tracker.format_per_joint_angle(val_metrics['per_joint_ang_err_deg'])}"
            )

            print(train_joint_angle_msg)
            print(val_joint_angle_msg)

            logger.record([train_joint_angle_msg], print_flag=False)
            logger.record([val_joint_angle_msg], print_flag=False)

            if train_metrics["per_joint_pos_err_cm"] is not None:
                train_joint_pos_msg = (
                    f"[{timestamp}] Epoch: {epoch}, train per-joint positional error (cm):\n"
                    f"{train_tracker.format_per_joint_pos(train_metrics['per_joint_pos_err_cm'])}"
                )
                print(train_joint_pos_msg)
                logger.record([train_joint_pos_msg], print_flag=False)

            if val_metrics["per_joint_pos_err_cm"] is not None:
                val_joint_pos_msg = (
                    f"[{timestamp}] Epoch: {epoch}, val per-joint positional error (cm):\n"
                    f"{train_tracker.format_per_joint_pos(val_metrics['per_joint_pos_err_cm'])}"
                )
                print(val_joint_pos_msg)
                logger.record([val_joint_pos_msg], print_flag=False)

            # ckpt_dir = os.path.join(output_save_path, timestamp, "checkpoints")
            # os.makedirs(ckpt_dir, exist_ok=True)
            # torch.save(model.module.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        if epoch - best_epoch >= 20:
            break

    logger.record([f"[{timestamp}] The best mean joint angle error occurred in epoch {best_epoch}"])

    if best_val_metrics is not None:
        best_angle_line = train_tracker.format_angle_metrics(
            best_val_metrics,
            label="best val mean joint angle error"
        )
        best_pos_line = train_tracker.format_pos_metrics(
            best_val_metrics,
            label="best val positional error"
        )

        logger.record([f"[{timestamp}] {best_angle_line}, {best_pos_line}"])

        logger.record([
            f"[{timestamp}] Best val per-joint angle error (deg):\n"
            f"{train_tracker.format_per_joint_angle(best_val_metrics['per_joint_ang_err_deg'])}"
        ], print_flag=False)

        if best_val_metrics["per_joint_pos_err_cm"] is not None:
            logger.record([
                f"[{timestamp}] Best val per-joint positional error (cm):\n"
                f"{train_tracker.format_per_joint_pos(best_val_metrics['per_joint_pos_err_cm'])}"
            ], print_flag=False)

    # return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{best_epoch}.pth")