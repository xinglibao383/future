import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def normalize(x, eps=1e-6):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True) + eps
    return (x - mean) / std


def evaluate_loss_mpjpe(model, dataloader):
    metric = Accumulator(4)
    device = next(iter(model.parameters())).device
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = normalize(x)

            x, y = x.to(device), y[:, :15, :, :2].to(device)
            y_hat = model(x)

            loss = criterion(y_hat, y)
            error = torch.norm(y_hat - y, dim=-1)

            metric.add(loss.item() * x.shape[0], x.shape[0], error.sum().item(), error.numel())

    return metric[0] / metric[1], metric[2] / metric[3]


def train(model, train_loader, val_loader, mask_ratio, lr, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss, min_val_mpjpe = float('inf'), float('inf')
    best_val_epoch = 0
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = normalize(x)

            mask = torch.rand_like(x) >= mask_ratio
            x = x * mask.float()

            optimizer.zero_grad()

            x, y = x.to(devices[0]), y[:, :15, :, :2].to(devices[0])
            y_hat = model(x)

            error = torch.norm(y_hat - y, dim=-1).mean()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            metric.add(loss.item() * x.shape[0], x.shape[0], error.sum().item(), error.numel())

            if i != 0 and i % 20 == 0:
                train_loss, train_mpjpe = metric[0] / metric[1], metric[2] / metric[3]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}, train mpjpe: {train_mpjpe:.4f}')

        train_loss, train_mpjpe = metric[0] / metric[1], metric[2] / metric[3]
        val_loss, val_mpjpe = evaluate_loss_mpjpe(model, val_loader)
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train mpjpe: {train_mpjpe:.4f}, val loss: {val_loss:.4f}, val mpjpe: {val_mpjpe:.4f}'])

        if val_loss < min_val_loss and val_mpjpe < min_val_mpjpe:
            min_val_loss, min_val_mpjpe = val_loss, val_mpjpe
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))

    logger.record([f"The best val model occurred in epoch {best_val_epoch}"])