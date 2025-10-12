import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def evaluate_loss_mpjpe(model, dataloader):
    metric = Accumulator(4)
    device = next(iter(model.parameters())).device
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            mean = x.mean(dim=(0, 2), keepdim=True)
            std = x.std(dim=(0, 2), keepdim=True) + 1e-6  # 防止除以0
            x = (x - mean) / std

            x, y = x.to(device), y[:, :15, :, :2].to(device)
            y_hat = model(x)

            loss = criterion(y_hat, y)
            error = torch.norm(y_hat - y, dim=-1)

            metric.add(loss.item() * x.shape[0], x.shape[0], error.sum().item(), error.numel())

    return metric[0] / metric[1], metric[2] / metric[3]


def train(model, train_loader, val_loader, mask_ratio, lr, num_epochs, devices, checkpoint_save_path, logger):
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss, min_val_mpjpe = float('inf'), float('inf')
    best_val_epoch = 0
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            mean = x.mean(dim=(0, 2), keepdim=True)
            std = x.std(dim=(0, 2), keepdim=True) + 1e-6  # 防止除以0
            x = (x - mean) / std

            mask = torch.rand_like(x) >= mask_ratio
            x = x * mask.float()

            optimizer.zero_grad()

            x, y = x.to(devices[0]), y[:, :15, :, :2].to(devices[0])
            y_hat = model(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            metric.add(loss.item() * batch_size, batch_size)

            if i != 0 and i % 20 == 0:
                train_loss = metric[0] / metric[1]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}')

        train_loss = metric[0] / metric[1]

        val_loss, val_mpjpe = evaluate_loss_mpjpe(model, val_loader)
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val mpjpe: {val_mpjpe:.4f}'])

        if val_loss < min_val_loss and val_mpjpe < min_val_mpjpe:
            min_val_loss, min_val_mpjpe = val_loss, val_mpjpe
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))

    logger.record([f"The best val model occurred in epoch {best_val_epoch}"])