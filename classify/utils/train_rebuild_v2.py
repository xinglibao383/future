import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from datetime import datetime
from utils.accumulator import Accumulator
from utils.dataloader import *
from classify.utils.confusion_matrix import compute_confusion_matrix


def evaluate(model, dataloader, loss_func, logger):
    criterion = nn.MSELoss(reduction='none') if loss_func == "mse" else nn.L1Loss(reduction='none')
    device = next(iter(model.parameters())).device
    positive_count, negative_count = 0, 0
    losses, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y1, y2) in enumerate(dataloader):
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x).mean(dim=(1, 2))
            positive_count += (y1 == 1).sum().item()
            negative_count += (y1 == 0).sum().item()
            losses.extend(loss.cpu().numpy())
            labels.extend(y1.cpu().numpy())
    # threshold = np.percentile(losses, (positive_count / (positive_count + negative_count) * 100))
    positive_losses = np.array([l for l, label in zip(losses, labels) if label==1])
    threshold = np.percentile(positive_losses, 95)  # 95% 的正样本重构误差低于阈值
    losses, labels = np.array(losses), np.array(labels)
    return ((losses <= threshold) == (labels == 1)).sum() / (positive_count + negative_count)


def train(model, train_loader, val_loader, loss_func, mask_ratio, lr, weight_decay, num_epochs, devices, output_save_path, logger, timestamp):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if loss_func == "mse" else nn.L1Loss()

    for epoch in range(num_epochs):
        metric = Accumulator(4)
        model.train()
        for i, (x, y1, y2) in enumerate(train_loader):
            optimizer.zero_grad()
            x_positive, x_negative = x[y1 == 1], x[y1 == 0]
            mask_positive, mask_negative = torch.rand_like(x_positive) >= mask_ratio, torch.rand_like(x_negative) >= mask_ratio
            x_positive, x_positive_masked = x_positive.to(devices[0]), (x_positive.clone() * mask_positive.float()).to(devices[0])
            x_negative, x_negative_masked = x_negative.to(devices[0]), (x_negative.clone() * mask_negative.float()).to(devices[0])
            x_positive_hat, x_negative_hat = model(x_positive_masked), model(x_negative_masked)
            loss_positive, loss_negative = criterion(x_positive_hat, x_positive), criterion(x_negative_hat, x_negative)
            loss_positive.backward()
            optimizer.step()
            metric.add(loss_positive.item() * x_positive.shape[0], x_positive.shape[0], loss_negative.item() * x_negative.shape[0], x_negative.shape[0])
            if i != 0 and i % 100 == 0:
                print(f'Epoch: {epoch}, iter: {i}, train loss positive: {metric[0] / metric[1]:.4f}, train loss negative: {metric[2] / metric[3]:.4f}')
        val_acc = evaluate(model, val_loader, loss_func, logger)
        logger.record([f'[{timestamp}] Epoch: {epoch}, train loss positive: {metric[0] / metric[1]:.4f}, train loss negative: {metric[2] / metric[3]:.4f}, val acc: {val_acc:.4f}'])
        