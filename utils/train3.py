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
    metric = Accumulator(8)
    device = next(iter(model.parameters())).device
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for x1, y1, x2, y2 in dataloader:
            batch_size = x1.shape[0]
            x1, x2 = normalize(x1), normalize(x2)
            x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)
            y1_hat, x2_hat, y2_hat = model(x1)

            error1, error2 = torch.norm(y1_hat - y1, dim=-1).mean(), torch.norm(y2_hat - y2, dim=-1).mean()
            loss1, loss2, loss3 = criterion(y1_hat, y1), criterion(x2_hat, x2), criterion(y2_hat, y2)
            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size, error1.sum().item(), error1.numel(), error2.sum().item(), error2.numel())
    
    return metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3], metric[4] / metric[5], metric[6] / metric[7]


def train(model, train_loader, val_loader, mask_ratio, lr, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        metric = Accumulator(8)
        model.train()
        for i, (x1, y1, x2, y2) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = x1.shape[0]
            x1, x2 = normalize(x1), normalize(x2)
            mask = torch.rand_like(x1) >= mask_ratio
            x1 = x1 * mask.float()
            x1, y1, x2, y2 = x1.to(devices[0]), y1.to(devices[0]), x2.to(devices[0]), y2.to(devices[0])
            y1_hat, x2_hat, y2_hat = model(x1)

            error1, error2 = torch.norm(y1_hat - y1, dim=-1).mean(), torch.norm(y2_hat - y2, dim=-1).mean()
            loss1, loss2, loss3 = criterion(y1_hat, y1), criterion(x2_hat, x2), criterion(y2_hat, y2)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, batch_size, error1.sum().item(), error1.numel(), error2.sum().item(), error2.numel())

            if i != 0 and i % 20 == 0:
                train_loss1, train_loss2, train_loss3, train_mpjpe1, train_mpjpe2 = metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3], metric[4] / metric[5], metric[6] / metric[7]
                print(f'Epoch: {epoch}, iter: {i}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train mpjpe1: {train_mpjpe1:.4f}, train mpjpe2: {train_mpjpe2:.4f}')

        train_loss1, train_loss2, train_loss3, train_mpjpe1, train_mpjpe2 = metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3], metric[4] / metric[5], metric[6] / metric[7]
        val_loss1, val_loss2, val_loss3, val_mpjpe1, val_mpjpe2 = evaluate_loss_mpjpe(model, val_loader)
        logger.record([f'Epoch: {epoch}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train mpjpe1: {train_mpjpe1:.4f}, train mpjpe2: {train_mpjpe2:.4f}'])
        logger.record([f'Epoch: {epoch},   val loss1: {val_loss1:.4f},   val loss2: {val_loss2:.4f},   val loss3: {val_loss3:.4f},   val mpjpe1: {val_mpjpe1:.4f},   val mpjpe2: {val_mpjpe2:.4f}'])

        torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))