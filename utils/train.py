import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def evaluate_loss(model, dataloader, criterion):
    metric = Accumulator(2)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            batch_size = x.shape[0]
            x, y = x[:, :, :, :2], y[:, :, :, :2]
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            metric.add(loss.item() * batch_size, batch_size)

    return metric[0] / metric[1]


def evaluate_mpjpe(model, dataloader):
    metric = Accumulator(2)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x[:, :, :, :2], y[:, :, :, :2]
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            error = torch.norm(y_hat - y, dim=-1)

            metric.add(error.sum().item(), error.numel())

    return metric[0] / metric[1]


def train(model, train_loader, val_loader, lr, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state_dict = model.state_dict()
    min_val_loss = float('inf')
    min_val_loss_epoch = 0

    for epoch in range(num_epochs):
        metric = Accumulator(2)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = y.shape[0]
            x, y = x[:, :, :, :2], y[:, :, :, :2]
            x, y = x.to(devices[0]), y.to(devices[0])

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss.item() * batch_size, batch_size)
                if i % 15 == 0:
                    train_loss = metric[0] / metric[1]
                    print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}')
        
        train_loss = metric[0] / metric[1]
        val_loss, val_mpjpe = evaluate_loss(model, val_loader, criterion), evaluate_mpjpe(model, val_loader)
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val mpjpe: {val_mpjpe:.4f}'])
        torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))
        if val_loss < min_val_loss:
            best_state_dict = model.state_dict()
            min_val_loss = val_loss
            min_val_loss_epoch = epoch

    logger.record([f"The best val loss occurred in the {min_val_loss_epoch} epoch"])
    torch.save(best_state_dict, os.path.join(checkpoint_save_path, "best_state_dict.pth"))