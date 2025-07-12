import os
import torch
import pandas as pd
from torch import nn
from utils.accumulator import Accumulator
from utils.tools import *
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(model, dataloader):
    device = next(iter(model.parameters())).device
    all_preds_y1, all_labels_y1 = [], []
    all_preds_y2, all_labels_y2 = [], []
    all_preds_y3, all_labels_y3 = [], []

    model.eval()
    with torch.no_grad():
        for x, y1, y2, y3 in dataloader:
            x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(device)
            y1_hat, y2_hat, y3_hat = model(x)
            all_preds_y1.extend(torch.argmax(y1_hat, dim=1).cpu().numpy())
            all_labels_y1.extend(y1.cpu().numpy())
            all_preds_y2.extend(torch.argmax(y2_hat, dim=1).cpu().numpy())
            all_labels_y2.extend(y2.cpu().numpy())
            all_preds_y3.extend(torch.argmax(y3_hat, dim=1).cpu().numpy())
            all_labels_y3.extend(y3.cpu().numpy())
    cm_y1 = confusion_matrix(all_labels_y1, all_preds_y1)
    cm_y2 = confusion_matrix(all_labels_y2, all_preds_y2)
    cm_y3 = confusion_matrix(all_labels_y3, all_preds_y3)

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum_y1 = cm_y1.sum(axis=1, keepdims=True)
        normalized_cm_y1 = np.divide(cm_y1, row_sum_y1, out=np.zeros_like(cm_y1, dtype=float), where=row_sum_y1 != 0)

        row_sum_y2 = cm_y2.sum(axis=1, keepdims=True)
        normalized_cm_y2 = np.divide(cm_y2, row_sum_y2, out=np.zeros_like(cm_y2, dtype=float), where=row_sum_y2 != 0)

        row_sum_y3 = cm_y3.sum(axis=1, keepdims=True)
        normalized_cm_y3 = np.divide(cm_y3, row_sum_y3, out=np.zeros_like(cm_y3, dtype=float), where=row_sum_y3 != 0)

    return np.round(normalized_cm_y1, 2), np.round(normalized_cm_y2, 2), np.round(normalized_cm_y3, 2)


def accuracy(y_hat, y):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def evaluate(model, dataloader, criterion):
    metric = Accumulator(8)
    device = next(iter(model.parameters())).device
    model.eval()
    with torch.no_grad():
        for i, (x, y1, y2, y3) in enumerate(dataloader):
            batch_size = x.shape[0]
            x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(device)
            y1_hat, y2_hat, y3_hat = model(x)
            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(y2_hat, y2)
            loss3 = criterion(y3_hat, y3)
            loss = loss1 + loss2 + loss3
            acc1 = accuracy(y1_hat, y1)
            acc2 = accuracy(y2_hat, y2)
            acc3 = accuracy(y3_hat, y3)
            metric.add(loss.item() * batch_size, loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, 
                       acc1 * batch_size, acc2 * batch_size, acc3 * batch_size,
                       batch_size)
        train_loss = metric[0] / metric[7]
        train_loss1 = metric[1] / metric[7]
        train_loss2 = metric[2] / metric[7]
        train_loss3 = metric[3] / metric[7]
        train_acc1 = metric[4] / metric[7]
        train_acc2 = metric[5] / metric[7]
        train_acc3 = metric[6] / metric[7]
    return train_loss, train_loss1, train_loss2, train_loss3, train_acc1, train_acc2, train_acc3


def train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, checkpoint_save_path, logger, alpha, beta, gamma):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        metric = Accumulator(8)
        model.train()
        for i, (x, y1, y2, y3) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x.shape[0]
            if mask_ratio > 0:
                mask = torch.rand_like(x) > mask_ratio
                x = x * mask.float()
            x, y1, y2, y3 = x.to(devices[0]), y1.to(devices[0]), y2.to(devices[0]), y3.to(devices[0])
            y1_hat, y2_hat, y3_hat = model(x)
            loss1 = criterion(y1_hat, y1)
            loss2 = criterion(y2_hat, y2)
            loss3 = criterion(y3_hat, y3)
            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            loss.backward()
            optimizer.step()
            acc1 = accuracy(y1_hat, y1)
            acc2 = accuracy(y2_hat, y2)
            acc3 = accuracy(y3_hat, y3)
            metric.add(loss.item() * batch_size, loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, 
                       acc1 * batch_size, acc2 * batch_size, acc3 * batch_size, batch_size)
            if i % 20 == 0:
                train_loss = metric[0] / metric[7]
                train_loss1 = metric[1] / metric[7]
                train_loss2 = metric[2] / metric[7]
                train_loss3 = metric[3] / metric[7]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}')
        
        train_loss = metric[0] / metric[7]
        train_loss1 = metric[1] / metric[7]
        train_loss2 = metric[2] / metric[7]
        train_loss3 = metric[3] / metric[7]
        train_acc1 = metric[4] / metric[7]
        train_acc2 = metric[5] / metric[7]
        train_acc3 = metric[6] / metric[7]
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train acc1: {train_acc1:.4f}, train acc2: {train_acc2:.4f}, train acc3: {train_acc3:.4f}'])
        val_loss, val_loss1, val_loss2, val_loss3, val_acc1, val_acc2, val_acc3 = evaluate(model, val_loader, criterion)
        logger.record([f'Epoch: {epoch}, val loss: {val_loss:.4f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, val loss3: {val_loss3:.4f}, val acc1: {val_acc1:.4f}, val acc2: {val_acc2:.4f}, val acc3: {val_acc3:.4f}'])
        train_confusion_matrix = compute_confusion_matrix(model, train_loader)
        logger.record([f'Epoch: {epoch} train confusion matrix1:\n{pd.DataFrame(train_confusion_matrix[0])}'], print_flag=False)
        logger.record([f'Epoch: {epoch} train confusion matrix2:\n{pd.DataFrame(train_confusion_matrix[1])}'], print_flag=False)
        logger.record([f'Epoch: {epoch} train confusion matrix3:\n{pd.DataFrame(train_confusion_matrix[2])}'], print_flag=False)
        val_confusion_matrix = compute_confusion_matrix(model, val_loader)
        logger.record([f'Epoch: {epoch} val confusion matrix1:\n{pd.DataFrame(val_confusion_matrix[0])}'], print_flag=False)
        logger.record([f'Epoch: {epoch} val confusion matrix2:\n{pd.DataFrame(val_confusion_matrix[1])}'], print_flag=False)
        logger.record([f'Epoch: {epoch} val confusion matrix3:\n{pd.DataFrame(val_confusion_matrix[2])}'], print_flag=False)
        if checkpoint_save_path != '':
            os.makedirs(checkpoint_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))