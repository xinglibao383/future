import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def normalize(x, eps=1e-6):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True) + eps
    return (x - mean) / std


def evaluate_loss_mpjpe(model, dataloader, criterion, need_normalize):
    metric = Accumulator(4)
    device = next(iter(model.parameters())).device
    model.eval()
    with torch.no_grad():
        for i, (x1, y1, z1, _, _, _) in enumerate(dataloader):
            batch_size = x1.shape[0]
            if need_normalize:
                x1 = normalize(x1)
            x1, y1, z1 = x1.to(device), y1.to(device), z1.to(device)
            y1_hat = model(x1)
            loss1 = criterion(y1_hat, y1)
            y1_hat, y1 = y1_hat.clamp(min=-0.9999, max=0.9999), y1.clamp(min=-0.9999, max=0.9999)
            y1_hat, y1 = torch.atanh(y1_hat), torch.atanh(y1)
            y1_hat, y1 = y1_hat * z1, y1 * z1
            error1 = torch.norm(y1_hat - y1, dim=-1).mean()
            metric.add(loss1.item() * batch_size, batch_size, error1.sum().item(), error1.numel())
    return metric[0] / metric[1], metric[2] / metric[3]


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()
    min_val_mpjpe, best_epoch = float('inf'), 0
    for epoch in range(num_epochs):
        metric = Accumulator(4)
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
            y1_hat, y1 = y1_hat.clamp(min=-0.9999, max=0.9999), y1.clamp(min=-0.9999, max=0.9999)
            y1_hat, y1 = torch.atanh(y1_hat), torch.atanh(y1)
            y1_hat, y1 = y1_hat * z1, y1 * z1
            error1 = torch.norm(y1_hat - y1, dim=-1).mean()
            metric.add(loss1.item() * batch_size, batch_size, error1.sum().item(), error1.numel())
            if i != 0 and i % 20 == 0:
                train_loss1, train_mpjpe1 = metric[0] / metric[1], metric[2] / metric[3]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss1:.4f}, train mpjpe: {train_mpjpe1:.4f}')
        train_loss1, train_mpjpe1 = metric[0] / metric[1], metric[2] / metric[3]
        val_loss1, val_mpjpe1 = evaluate_loss_mpjpe(model, val_loader, criterion, need_normalize)
        logger.record([f'[{timestamp}] Epoch: {epoch}, train loss: {train_loss1:.4f}, train mpjpe: {train_mpjpe1:.4f}'])
        logger.record([f'[{timestamp}] Epoch: {epoch},   val loss: {val_loss1:.4f},   val mpjpe: {val_mpjpe1:.4f}'])
        if val_mpjpe1 < min_val_mpjpe:
            min_val_mpjpe = val_mpjpe1
            best_epoch = epoch
        if epoch - best_epoch >= 20:
            break
    logger.record([f'[{timestamp}] The best mpjpe occurred in epoch {best_epoch}'])
    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{epoch}.pth")