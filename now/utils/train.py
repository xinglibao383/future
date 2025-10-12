import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.confusion_matrix import compute_confusion_matrix


def accuracy(y_hat, y):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def evaluate(model, dataloader, criterion1, criterion2):
    metric = Accumulator(3)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion1(y_hat, y)
            acc = accuracy(y_hat, y)
            metric.add(loss.item() * batch_size, acc * batch_size, batch_size)
        
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
    return train_loss, train_acc


def train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x.shape[0]
            if mask_ratio > 0:
                mask = torch.rand_like(x) > mask_ratio
                x = x * mask.float()
            x, y = x.to(devices[0]), y.to(devices[0])
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            acc = accuracy(y_hat, y)

            metric.add(loss.item() * batch_size, acc * batch_size, batch_size)

            if i % 20 == 0:
                train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        
        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc1:.4f}'])
        
        if train_acc1 > 0.8 and train_acc2 > 0.3 and (train_acc1 > max_train_acc1 and train_acc2 > max_train_acc2 or val_acc1 > max_val_acc1 and val_acc2 > max_val_acc2):
            logger.record([f'Epoch: {epoch} train confusion matrix:\n{pd.DataFrame(compute_confusion_matrix(model, train_loader))}'], print_flag=False)
            logger.record([f'Epoch: {epoch} val confusion matrix:\n{pd.DataFrame(compute_confusion_matrix(model, val_loader))}'], print_flag=False)
        if save_flag and checkpoint_save_path != '':
            os.makedirs(checkpoint_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))