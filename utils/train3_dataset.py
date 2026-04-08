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


def summarize_two_trackers(current_tracker, future_tracker):
    current_metrics = current_tracker.summary()
    future_metrics = future_tracker.summary()

    metrics = {
        "current_ang_err_deg": current_metrics["ang_err_deg"],
        "current_per_joint_ang_err_deg": current_metrics["per_joint_ang_err_deg"],
        "current_pos_err_cm": current_metrics["pos_err_cm"],
        "current_per_joint_pos_err_cm": current_metrics["per_joint_pos_err_cm"],

        "future_ang_err_deg": future_metrics["ang_err_deg"],
        "future_per_joint_ang_err_deg": future_metrics["per_joint_ang_err_deg"],
        "future_pos_err_cm": future_metrics["pos_err_cm"],
        "future_per_joint_pos_err_cm": future_metrics["per_joint_pos_err_cm"],
    }
    return metrics


def evaluate_loss_dip(model, dataloader, criterion, need_normalize, smpl_forward=None):
    metric = Accumulator(4)
    device = next(iter(model.parameters())).device
    model.eval()

    current_tracker = DIPMetricTracker()
    future_tracker = DIPMetricTracker()

    with torch.no_grad():
        for _, (x1, y1, x2, y2) in enumerate(dataloader):
            batch_size = x1.shape[0]

            x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
            x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
            y1 = torch.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)
            y2 = torch.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)

            if need_normalize:
                x1 = normalize(x1)
                x2 = normalize(x2)

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            y1_hat, x2_hat, y2_hat = model(x1)

            y1_hat = torch.nan_to_num(y1_hat, nan=0.0, posinf=0.0, neginf=0.0)
            x2_hat = torch.nan_to_num(x2_hat, nan=0.0, posinf=0.0, neginf=0.0)
            y2_hat = torch.nan_to_num(y2_hat, nan=0.0, posinf=0.0, neginf=0.0)

            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(x2_hat, x2)
            loss3 = criterion(y2_hat, y2)
            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size)

            current_tracker.update_angle(y1_hat, y1)
            future_tracker.update_angle(y2_hat, y2)

            if smpl_forward is not None:
                pred_current_joints = smpl_forward.pose_to_joints(y1_hat)[..., :24, :]
                gt_current_joints = smpl_forward.pose_to_joints(y1)[..., :24, :]
                current_tracker.update_pos(pred_current_joints, gt_current_joints)

                pred_future_joints = smpl_forward.pose_to_joints(y2_hat)[..., :24, :]
                gt_future_joints = smpl_forward.pose_to_joints(y2)[..., :24, :]
                future_tracker.update_pos(pred_future_joints, gt_future_joints)

    return {
        "loss1": metric[0] / metric[3],
        "loss2": metric[1] / metric[3],
        "loss3": metric[2] / metric[3],
        **summarize_two_trackers(current_tracker, future_tracker),
    }


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize,
          alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp):
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
        metric = Accumulator(4)
        current_tracker = DIPMetricTracker()
        future_tracker = DIPMetricTracker()
        model.train()

        for i, (x1, y1, x2, y2) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x1.shape[0]

            x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
            x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
            y1 = torch.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)
            y2 = torch.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)

            if need_normalize:
                x1 = normalize(x1)
                x2 = normalize(x2)

            mask = (torch.rand_like(x1) >= mask_ratio).float()
            x1 = x1 * mask

            x1 = x1.to(devices[0], non_blocking=True)
            x2 = x2.to(devices[0], non_blocking=True)
            y1 = y1.to(devices[0], non_blocking=True)
            y2 = y2.to(devices[0], non_blocking=True)

            y1_hat, x2_hat, y2_hat = model(x1)

            y1_hat = torch.nan_to_num(y1_hat, nan=0.0, posinf=0.0, neginf=0.0)
            x2_hat = torch.nan_to_num(x2_hat, nan=0.0, posinf=0.0, neginf=0.0)
            y2_hat = torch.nan_to_num(y2_hat, nan=0.0, posinf=0.0, neginf=0.0)

            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(x2_hat, x2)
            loss3 = criterion(y2_hat, y2)
            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            loss.backward()
            optimizer.step()

            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size)

            current_tracker.update_angle(y1_hat.detach(), y1)
            future_tracker.update_angle(y2_hat.detach(), y2)

            if smpl_forward is not None:
                pred_current_joints = smpl_forward.pose_to_joints(y1_hat.detach())[..., :24, :]
                gt_current_joints = smpl_forward.pose_to_joints(y1)[..., :24, :]
                current_tracker.update_pos(pred_current_joints, gt_current_joints)

                pred_future_joints = smpl_forward.pose_to_joints(y2_hat.detach())[..., :24, :]
                gt_future_joints = smpl_forward.pose_to_joints(y2)[..., :24, :]
                future_tracker.update_pos(pred_future_joints, gt_future_joints)

            if i != 0 and i % 20 == 0:
                train_metrics = summarize_two_trackers(current_tracker, future_tracker)
                train_loss1 = metric[0] / metric[3]
                train_loss2 = metric[1] / metric[3]
                train_loss3 = metric[2] / metric[3]

                current_angle_line = current_tracker.format_angle_metrics(
                    {"ang_err_deg": train_metrics["current_ang_err_deg"]},
                    label="train current mean joint angle error"
                )
                current_pos_line = current_tracker.format_pos_metrics(
                    {"pos_err_cm": train_metrics["current_pos_err_cm"]},
                    label="train current positional error"
                )
                future_angle_line = future_tracker.format_angle_metrics(
                    {"ang_err_deg": train_metrics["future_ang_err_deg"]},
                    label="train future mean joint angle error"
                )
                future_pos_line = future_tracker.format_pos_metrics(
                    {"pos_err_cm": train_metrics["future_pos_err_cm"]},
                    label="train future positional error"
                )

                print(
                    f"Epoch: {epoch}, iter: {i}, "
                    f"train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, "
                    f"{current_angle_line}, {current_pos_line}, "
                    f"{future_angle_line}, {future_pos_line}"
                )

        train_metrics = summarize_two_trackers(current_tracker, future_tracker)
        train_loss1 = metric[0] / metric[3]
        train_loss2 = metric[1] / metric[3]
        train_loss3 = metric[2] / metric[3]

        val_metrics = evaluate_loss_dip(model, val_loader, criterion, need_normalize, smpl_forward)

        train_current_angle_line = current_tracker.format_angle_metrics(
            {"ang_err_deg": train_metrics["current_ang_err_deg"]},
            label="train current mean joint angle error"
        )
        train_current_pos_line = current_tracker.format_pos_metrics(
            {"pos_err_cm": train_metrics["current_pos_err_cm"]},
            label="train current positional error"
        )
        train_future_angle_line = future_tracker.format_angle_metrics(
            {"ang_err_deg": train_metrics["future_ang_err_deg"]},
            label="train future mean joint angle error"
        )
        train_future_pos_line = future_tracker.format_pos_metrics(
            {"pos_err_cm": train_metrics["future_pos_err_cm"]},
            label="train future positional error"
        )

        val_current_angle_line = current_tracker.format_angle_metrics(
            {"ang_err_deg": val_metrics["current_ang_err_deg"]},
            label="val current mean joint angle error"
        )
        val_current_pos_line = current_tracker.format_pos_metrics(
            {"pos_err_cm": val_metrics["current_pos_err_cm"]},
            label="val current positional error"
        )
        val_future_angle_line = future_tracker.format_angle_metrics(
            {"ang_err_deg": val_metrics["future_ang_err_deg"]},
            label="val future mean joint angle error"
        )
        val_future_pos_line = future_tracker.format_pos_metrics(
            {"pos_err_cm": val_metrics["future_pos_err_cm"]},
            label="val future positional error"
        )

        train_msg = (
            f"[{timestamp}] Epoch: {epoch}, "
            f"train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, "
            f"{train_current_angle_line}, {train_current_pos_line}, "
            f"{train_future_angle_line}, {train_future_pos_line}"
        )
        val_msg = (
            f"[{timestamp}] Epoch: {epoch}, "
            f"val loss1: {val_metrics['loss1']:.4f}, val loss2: {val_metrics['loss2']:.4f}, val loss3: {val_metrics['loss3']:.4f}, "
            f"{val_current_angle_line}, {val_current_pos_line}, "
            f"{val_future_angle_line}, {val_future_pos_line}"
        )

        current_score = val_metrics["current_ang_err_deg"] + val_metrics["future_ang_err_deg"]
        is_best_epoch = current_score < min_val_ang_err
        if is_best_epoch:
            min_val_ang_err = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics

        print(train_msg)
        print(val_msg)
        logger.record([train_msg])
        logger.record([val_msg])

        if is_best_epoch:
            train_current_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, train current per-joint angle error (deg):\n"
                f"{current_tracker.format_per_joint_angle(train_metrics['current_per_joint_ang_err_deg'])}"
            )
            train_future_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, train future per-joint angle error (deg):\n"
                f"{future_tracker.format_per_joint_angle(train_metrics['future_per_joint_ang_err_deg'])}"
            )
            val_current_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, val current per-joint angle error (deg):\n"
                f"{current_tracker.format_per_joint_angle(val_metrics['current_per_joint_ang_err_deg'])}"
            )
            val_future_joint_angle_msg = (
                f"[{timestamp}] Epoch: {epoch}, val future per-joint angle error (deg):\n"
                f"{future_tracker.format_per_joint_angle(val_metrics['future_per_joint_ang_err_deg'])}"
            )

            print(train_current_joint_angle_msg)
            print(train_future_joint_angle_msg)
            print(val_current_joint_angle_msg)
            print(val_future_joint_angle_msg)

            logger.record([train_current_joint_angle_msg], print_flag=False)
            logger.record([train_future_joint_angle_msg], print_flag=False)
            logger.record([val_current_joint_angle_msg], print_flag=False)
            logger.record([val_future_joint_angle_msg], print_flag=False)

            if train_metrics["current_per_joint_pos_err_cm"] is not None:
                msg = (
                    f"[{timestamp}] Epoch: {epoch}, train current per-joint positional error (cm):\n"
                    f"{current_tracker.format_per_joint_pos(train_metrics['current_per_joint_pos_err_cm'])}"
                )
                print(msg)
                logger.record([msg], print_flag=False)

            if train_metrics["future_per_joint_pos_err_cm"] is not None:
                msg = (
                    f"[{timestamp}] Epoch: {epoch}, train future per-joint positional error (cm):\n"
                    f"{future_tracker.format_per_joint_pos(train_metrics['future_per_joint_pos_err_cm'])}"
                )
                print(msg)
                logger.record([msg], print_flag=False)

            if val_metrics["current_per_joint_pos_err_cm"] is not None:
                msg = (
                    f"[{timestamp}] Epoch: {epoch}, val current per-joint positional error (cm):\n"
                    f"{current_tracker.format_per_joint_pos(val_metrics['current_per_joint_pos_err_cm'])}"
                )
                print(msg)
                logger.record([msg], print_flag=False)

            if val_metrics["future_per_joint_pos_err_cm"] is not None:
                msg = (
                    f"[{timestamp}] Epoch: {epoch}, val future per-joint positional error (cm):\n"
                    f"{future_tracker.format_per_joint_pos(val_metrics['future_per_joint_pos_err_cm'])}"
                )
                print(msg)
                logger.record([msg], print_flag=False)

            # ckpt_dir = os.path.join(output_save_path, timestamp, "checkpoints")
            # os.makedirs(ckpt_dir, exist_ok=True)
            # torch.save(model.module.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        if epoch - best_epoch >= 20:
            break

    logger.record([f"[{timestamp}] The best mean joint angle error occurred in epoch {best_epoch}"])

    if best_val_metrics is not None:
        best_current_angle_line = current_tracker.format_angle_metrics(
            {"ang_err_deg": best_val_metrics["current_ang_err_deg"]},
            label="best val current mean joint angle error"
        )
        best_current_pos_line = current_tracker.format_pos_metrics(
            {"pos_err_cm": best_val_metrics["current_pos_err_cm"]},
            label="best val current positional error"
        )
        best_future_angle_line = future_tracker.format_angle_metrics(
            {"ang_err_deg": best_val_metrics["future_ang_err_deg"]},
            label="best val future mean joint angle error"
        )
        best_future_pos_line = future_tracker.format_pos_metrics(
            {"pos_err_cm": best_val_metrics["future_pos_err_cm"]},
            label="best val future positional error"
        )

        logger.record([
            f"[{timestamp}] "
            f"{best_current_angle_line}, {best_current_pos_line}, "
            f"{best_future_angle_line}, {best_future_pos_line}"
        ])

        logger.record([
            f"[{timestamp}] Best val current per-joint angle error (deg):\n"
            f"{current_tracker.format_per_joint_angle(best_val_metrics['current_per_joint_ang_err_deg'])}"
        ], print_flag=False)
        logger.record([
            f"[{timestamp}] Best val future per-joint angle error (deg):\n"
            f"{future_tracker.format_per_joint_angle(best_val_metrics['future_per_joint_ang_err_deg'])}"
        ], print_flag=False)

        if best_val_metrics["current_per_joint_pos_err_cm"] is not None:
            logger.record([
                f"[{timestamp}] Best val current per-joint positional error (cm):\n"
                f"{current_tracker.format_per_joint_pos(best_val_metrics['current_per_joint_pos_err_cm'])}"
            ], print_flag=False)

        if best_val_metrics["future_per_joint_pos_err_cm"] is not None:
            logger.record([
                f"[{timestamp}] Best val future per-joint positional error (cm):\n"
                f"{future_tracker.format_per_joint_pos(best_val_metrics['future_per_joint_pos_err_cm'])}"
            ], print_flag=False)

    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{best_epoch}.pth")