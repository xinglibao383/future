import os
import torch
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *


def evaluate_loss(model, dataloader, criterion):
    metric = Accumulator(6)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for x_imu, y_imu, x_pose, y_pose in dataloader:
            batch_size = x_imu.shape[0]
            x_imu, y_imu, x_pose, y_pose = x_imu.to(device), y_imu.to(device), x_pose.to(device), y_pose.to(device)
            pose, future_pose1, future_imu, future_pose2 = model(x_imu)
            loss1 = criterion(x_pose, pose)
            loss2 = criterion(y_pose, future_pose1)
            loss3 = criterion(y_imu, future_imu)
            loss4 = criterion(y_pose, future_pose2)
            # loss = loss1 + loss2 + loss3 + loss4
            loss = loss1
            metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, loss4.item() * batch_size, loss.item() * batch_size, batch_size)

    return metric[0] / metric[5], metric[1] / metric[5], metric[2] / metric[5], metric[3] / metric[5], metric[4] / metric[5]


def train(model, train_loader, val_loader, lr, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss = float('inf')
    min_val_loss_epoch = 0

    for epoch in range(num_epochs):
        metric = Accumulator(6)
        model.train()
        for i, (x_imu, y_imu, x_pose, y_pose) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = x_imu.shape[0]
            x_imu, y_imu, x_pose, y_pose = x_imu.to(devices[0]), y_imu.to(devices[0]), x_pose.to(devices[0]), y_pose.to(devices[0])
            pose, future_pose1, future_imu, future_pose2 = model(x_imu)
            loss1 = criterion(x_pose, pose)
            loss2 = criterion(y_pose, future_pose1)
            loss3 = criterion(y_imu, future_imu)
            loss4 = criterion(y_pose, future_pose2)
            # loss = loss1 + loss2 + loss3 + loss4
            loss = loss1
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, loss4.item() * batch_size, loss.item() * batch_size, batch_size)
                if i % 20 == 0:
                    train_loss1 = metric[0] / metric[5]
                    train_loss2 = metric[1] / metric[5]
                    train_loss3 = metric[2] / metric[5]
                    train_loss4 = metric[3] / metric[5]
                    train_loss = metric[4] / metric[5]
                    print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train loss4: {train_loss4:.4f}')
        
        train_loss1 = metric[0] / metric[5]
        train_loss2 = metric[1] / metric[5]
        train_loss3 = metric[2] / metric[5]
        train_loss4 = metric[3] / metric[5]
        train_loss = metric[4] / metric[5]
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train loss4: {train_loss4:.4f}'])
        val_loss1, val_loss2, val_loss3, val_loss4, val_loss = evaluate_loss(model, val_loader, criterion)
        logger.record([f'Epoch: {epoch}, val loss: {val_loss:.4f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, val loss3: {val_loss3:.4f}, val loss4: {val_loss4:.4f}'])
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))

    logger.record([f"The best val loss occurred in the {min_val_loss_epoch} epoch"])