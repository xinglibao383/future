import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *

# ===== 1. 评估 MSE loss =====
def evaluate_loss(model, dataloader):
    metric = Accumulator(2)  # [sum_loss, num_batches]
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            mean = x.mean(dim=(0, 2), keepdim=True)   # 按 batch 和时间轴求均值
            std = x.std(dim=(0, 2), keepdim=True) + 1e-6  # 防止除以0
            x = (x - mean) / std
            

            x, y = x.to(device), y[:, :, :2].to(device)  # 只取关键点坐标
            y_hat = model(x)

            loss = ((y_hat - y) ** 2).mean()  # MSE over all elements

            batch_size = x.shape[0]
            metric.add(loss.item() * batch_size, batch_size)  # 按 batch 累加

    return metric[0] / metric[1]


# ===== 2. 评估 MPJPE =====
def evaluate_mpjpe(model, dataloader):
    metric = Accumulator(2)  # [sum_error, total_count]
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            mean = x.mean(dim=(0, 2), keepdim=True)   # 按 batch 和时间轴求均值
            std = x.std(dim=(0, 2), keepdim=True) + 1e-6  # 防止除以0
            x = (x - mean) / std

            x, y = x.to(device), y[:, :, :2].to(device)
            y_hat = model(x)

            error = torch.norm(y_hat - y, dim=-1)  # [B, num_kp]
            batch_size = x.shape[0]
            num_kp = y.shape[1]

            metric.add(error.sum().item(), batch_size * num_kp)

    return metric[0] / metric[1]


# ===== 3. 训练函数 =====
def train(model, train_loader, val_loader, lr, num_epochs, devices, checkpoint_save_path, logger):
    # 权重初始化
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    # model.apply(init_weights)

    # 支持多 GPU
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss = float('inf')
    min_val_loss_epoch = 0

    for epoch in range(num_epochs):
        metric = Accumulator(2)  # [sum_loss, total_samples]
        model.train()
        for i, (x, y) in enumerate(train_loader):

            mean = x.mean(dim=(0, 2), keepdim=True)   # 按 batch 和时间轴求均值
            std = x.std(dim=(0, 2), keepdim=True) + 1e-6  # 防止除以0
            x = (x - mean) / std

            mask = torch.rand_like(x) >= 0.25
            x = x * mask.float()


            optimizer.zero_grad()

            x, y = x.to(devices[0]), y[:, :, :2].to(devices[0])
            y_hat = model(x)

            loss = criterion(y_hat, y)  # standard MSE
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            metric.add(loss.item() * batch_size, batch_size)

            if i % 20 == 0:
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