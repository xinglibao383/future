import os
import torch
import shutil
import datetime
import random
import matplotlib.pyplot as plt
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def normalize(x, eps=1e-6):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True) + eps
    return (x - mean) / std


def plot_poses(data, output_save_path, timestamp):
    def plot_pose(pose, save_filepath):
        x, y = pose[:,0].numpy(), pose[:,1].numpy()
        plt.figure(figsize=(8,10))
        plt.scatter(x, y, c='red', s=50)
        for i,(xi, yi) in enumerate(zip(x, y)):
            plt.text(xi+0.02, yi+0.02, f"({xi:.2f},{yi:.2f})", fontsize=8, color='blue')
        skeleton = [
            (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
            (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
            (0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(14,21),
            (11,22),(22,23),(11,24)
        ]
        for i,j in skeleton:
            plt.plot([x[i],x[j]], [y[i],y[j]], 'g-', linewidth=2)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        plt.close()

    img_save_path = os.path.join(output_save_path, timestamp, "imgs")
    shutil.rmtree(img_save_path)
    os.makedirs(img_save_path, exist_ok=True)
    poses = data.clone().cpu().reshape(-1, 25, 2)
    poses = poses.clamp(min=-0.9999, max=0.9999)
    poses = torch.atanh(poses)
    idxs = torch.randperm(poses.shape[0])[:10]
    for idx in idxs:
        if torch.isfinite(poses[idx]).all():
            plot_pose(poses[idx], os.path.join(img_save_path, f"{idx}.png"))


def evaluate_loss_mpjpe(model, dataloader, criterion, need_normalize, timestamp, output_save_path):
    metric = Accumulator(6)
    device = next(iter(model.parameters())).device
    model.eval()
    iterIdx = random.randint(0, len(dataloader))
    with torch.no_grad():
        for i, (x, y, z) in enumerate(dataloader):
            batch_size = x.shape[0]
            if need_normalize:
                x = normalize(x)
            x, y, z = x.to(device), y.to(device), z.to(device)
            y_hat = model(x)

            if i == iterIdx: plot_poses(y_hat, output_save_path, timestamp)

            loss1, loss2 = criterion(y_hat[:, :15, :, :], y[:, :15, :, :]), criterion(y_hat[:, 15:, :, :], y[:, 15:, :, :])
            y_hat, y = y_hat.clamp(min=-0.9999, max=0.9999), y.clamp(min=-0.9999, max=0.9999)
            y_hat, y = torch.atanh(y_hat), torch.atanh(y)
            y_hat, y = y_hat * z, y * z
            error1, error2 = torch.norm(y_hat[:, :15, :, :] - y[:, :15, :, :], dim=-1).mean(), torch.norm(y_hat[:, 15:, :, :] - y[:, 15:, :, :], dim=-1).mean()

            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, batch_size, error1.sum().item(), error2.sum().item(), error1.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2], metric[3] / metric[5], metric[4] / metric[5]


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, need_normalize, alpha, beta, gamma, num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()

    min_val_mpjpe, best_epoch = float('inf'), 0

    for epoch in range(num_epochs):
        metric = Accumulator(6)
        model.train()
        for i, (x, y, z) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = x.shape[0]
            if need_normalize:
                x = normalize(x)
            mask = torch.rand_like(x) >= mask_ratio
            x = x * mask.float()
            x, y, z = x.to(devices[0]), y.to(devices[0]), z.to(devices[0])
            y_hat = model(x)

            loss1, loss2 = criterion(y_hat[:, :15, :, :], y[:, :15, :, :]), criterion(y_hat[:, 15:, :, :], y[:, 15:, :, :])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            y_hat, y = y_hat.clamp(min=-0.9999, max=0.9999), y.clamp(min=-0.9999, max=0.9999)
            y_hat, y = torch.atanh(y_hat), torch.atanh(y)
            y_hat, y = y_hat * z, y * z
            error1, error2 = torch.norm(y_hat[:, :15, :, :] - y[:, :15, :, :], dim=-1).mean(), torch.norm(y_hat[:, 15:, :, :] - y[:, 15:, :, :], dim=-1).mean()

            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, batch_size, error1.sum().item(), error2.sum().item(), error1.numel())

            if i != 0 and i % 20 == 0:
                train_loss1, train_loss2, train_mpjpe1, train_mpjpe2 = metric[0] / metric[2], metric[1] / metric[2], metric[3] / metric[5], metric[4] / metric[5]
                print(f'Epoch: {epoch}, iter: {i}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train mpjpe1: {train_mpjpe1:.4f}, train mpjpe2: {train_mpjpe2:.4f}')

        train_loss1, train_loss2, train_mpjpe1, train_mpjpe2 = metric[0] / metric[2], metric[1] / metric[2], metric[3] / metric[5], metric[4] / metric[5]
        val_loss1, val_loss2, val_mpjpe1, val_mpjpe2 = evaluate_loss_mpjpe(model, val_loader, criterion, need_normalize, timestamp, output_save_path)
        logger.record([f'[{timestamp}] Epoch: {epoch}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train mpjpe1: {train_mpjpe1:.4f}, train mpjpe2: {train_mpjpe2:.4f}'])
        logger.record([f'[{timestamp}] Epoch: {epoch},   val loss1: {val_loss1:.4f},   val loss2: {val_loss2:.4f},   val mpjpe1: {val_mpjpe1:.4f},   val mpjpe2: {val_mpjpe2:.4f}'])

        if val_mpjpe1 + val_mpjpe2 < min_val_mpjpe:
            min_val_mpjpe = val_mpjpe1 + val_mpjpe2
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{epoch}.pth"))
        if epoch - best_epoch >= 20:
            break
    
    logger.record([f'[{timestamp}] The best mpjpe occurred in epoch {best_epoch}'])
    return os.path.join(output_save_path, timestamp, "checkpoints", f"epoch_{epoch}.pth")