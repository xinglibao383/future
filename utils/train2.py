import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *

# ===== 1. 评估加权 MSE loss =====
def evaluate_loss(model, dataloader):
    metric = Accumulator(2)  # [sum_loss, total_weight]
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            batch_size = x.shape[0]
            # y: [B, num_kp, 2+?], confidence: last channel
            y_coords, confidence = y[:, :, :2], y[:, :, 2:]
            x, y_coords, confidence = x.to(device), y_coords.to(device), confidence.to(device)

            y_hat = model(x)  # [B, num_kp, 2]

            squared_error = ((y_hat - y_coords) ** 2).sum(dim=-1)  # [B, num_kp]
            weights = confidence.view(batch_size, -1)              # [B, num_kp]

            weighted_loss = (squared_error * weights).sum()
            total_weight = weights.sum()

            metric.add(weighted_loss.item(), total_weight.item())

    return metric[0] / metric[1]


# ===== 2. 评估加权 MPJPE =====
def evaluate_mpjpe(model, dataloader):
    metric = Accumulator(2)  # [weighted_error_sum, total_weight]
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            batch_size = x.shape[0]
            y_coords, confidence = y[:, :, :2], y[:, :, 2:]
            x, y_coords, confidence = x.to(device), y_coords.to(device), confidence.to(device)

            y_hat = model(x)  # [B, num_kp, 2]

            error = torch.norm(y_hat - y_coords, dim=-1)   # [B, num_kp]
            weights = confidence.view(batch_size, -1)      # [B, num_kp]

            weighted_error = (error * weights).sum()
            total_weight = weights.sum()

            metric.add(weighted_error.item(), total_weight.item())

    return metric[0] / metric[1]


# ===== 3. 训练函数 =====
def train(model, train_loader, val_loader, lr, num_epochs, devices, checkpoint_save_path, logger):
    # 权重初始化
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    # 支持多 GPU
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    min_val_loss = float('inf')
    min_val_loss_epoch = 0

    for epoch in range(num_epochs):
        metric = Accumulator(2)  # [weighted_loss_sum, total_weight]
        model.train()
        for i, (x, y) in enumerate(train_loader):
            print(x.shape, y.shape)
            optimizer.zero_grad()
            batch_size = x.shape[0]

            y_coords, confidence = y[:, :, :2], y[:, :, 2:]
            x, y_coords, confidence = x.to(devices[0]), y_coords.to(devices[0]), confidence.to(devices[0])

            y_hat = model(x)  # [B, num_kp, 2]

            squared_error = ((y_hat - y_coords) ** 2).sum(dim=-1)  # [B, num_kp]
            weights = confidence.view(batch_size, -1)              # [B, num_kp]

            loss = (squared_error * weights).sum() / weights.sum()  # 加权平均
            loss.backward()
            optimizer.step()

            # 累计训练 loss
            with torch.no_grad():
                weighted_loss = (squared_error * weights).sum()
                total_weight = weights.sum()
                metric.add(weighted_loss.item(), total_weight.item())

                if i != 0 and i % 100 == 0:
                    train_loss = metric[0] / metric[1]
                    print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}')

        # 每个 epoch 完成后计算 train loss
        train_loss = metric[0] / metric[1]

        # 验证集 loss & MPJPE
        val_loss = evaluate_loss(model, val_loader)
        val_mpjpe = evaluate_mpjpe(model, val_loader)

        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val mpjpe: {val_mpjpe:.4f}'])

        # 保存最优模型
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = epoch
            # torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))

    logger.record([f"The best val loss occurred in epoch {min_val_loss_epoch}"])