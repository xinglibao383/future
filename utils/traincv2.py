import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from utils.tools import *
from utils.confusion_matrix import compute_confusion_matrix


def accuracy(y_hat, y):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def evaluate(model, dataloader, criterion1, criterion2):
    metric = Accumulator(7)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for i, (x1, y1, x2, y2) in enumerate(dataloader):
            batch_size = x1.shape[0]
            x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)
            y1_hat, x2_hat, y2_hat = model(x1)
            loss1 = criterion1(y1_hat, y1)
            loss2 = criterion2(x2_hat, x2)
            loss3 = criterion1(y2_hat, y2)
            loss = loss1 + loss2 + loss3
            acc1 = accuracy(y1_hat, y1)
            acc2 = accuracy(y2_hat, y2)
            metric.add(loss.item() * batch_size, loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, 
                       acc1 * batch_size, acc2 * batch_size,
                       batch_size)
        
        train_loss = metric[0] / metric[6]
        train_loss1 = metric[1] / metric[6]
        train_loss2 = metric[2] / metric[6]
        train_loss3 = metric[3] / metric[6]
        train_acc1 = metric[4] / metric[6]
        train_acc2 = metric[5] / metric[6]
    return train_loss, train_loss1, train_loss2, train_loss3, train_acc1, train_acc2


def train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, checkpoint_save_path, logger, alpha, beta, gamma, max_invalid_num_epochs = -1, use_dynamic_weights = None):
    y1_weights, _, y1_y2_weights = get_class_weights(train_loader, val_loader, logger)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion1 = nn.CrossEntropyLoss()
    if use_dynamic_weights != None:
        if use_dynamic_weights == 'y1':
            y1_weights = y1_weights.to(devices[0])
            criterion1 = nn.CrossEntropyLoss(weight=y1_weights)
        elif use_dynamic_weights == 'y1 + y2':
            y1_y2_weights = y1_y2_weights.to(devices[0])
            criterion1 = nn.CrossEntropyLoss(weight=y1_y2_weights)
    criterion2 = nn.MSELoss()

    min_val_loss1 = float('inf')
    min_val_loss2 = float('inf')
    min_val_loss3 = float('inf')
    invalid_num_epochs = 0

    for epoch in range(num_epochs):
        metric = Accumulator(7)
        model.train()
        for i, (x1, y1, x2, y2) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = x1.shape[0]
            if mask_ratio > 0:
                mask = torch.rand_like(x1) > mask_ratio
                x1 = x1 * mask.float()
            x1, y1, x2, y2 = x1.to(devices[0]), y1.to(devices[0]), x2.to(devices[0]), y2.to(devices[0])
            y1_hat, x2_hat, y2_hat = model(x1)
            loss1 = criterion1(y1_hat, y1)
            loss2 = criterion2(x2_hat, x2)
            loss3 = criterion1(y2_hat, y2)
            # todo imu预测的权重可以低一些，毕竟只需要能识别出来在做什么动作即可
            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            loss.backward()
            optimizer.step()
            acc1 = accuracy(y1_hat, y1)
            acc2 = accuracy(y2_hat, y2)

            metric.add(loss.item() * batch_size, loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size, 
                       acc1 * batch_size, acc2 * batch_size,
                       batch_size)

            if i % 20 == 0:
                train_loss = metric[0] / metric[6]
                train_loss1 = metric[1] / metric[6]
                train_loss2 = metric[2] / metric[6]
                train_loss3 = metric[3] / metric[6]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}')
        
        train_loss = metric[0] / metric[6]
        train_loss1 = metric[1] / metric[6]
        train_loss2 = metric[2] / metric[6]
        train_loss3 = metric[3] / metric[6]
        train_acc1 = metric[4] / metric[6]
        train_acc2 = metric[5] / metric[6]
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train acc1: {train_acc1:.4f}, train acc2: {train_acc2:.4f}'])
        
        val_loss, val_loss1, val_loss2, val_loss3, val_acc1, val_acc2 = evaluate(model, val_loader, criterion1, criterion2)
        logger.record([f'Epoch: {epoch}, val loss: {val_loss:.4f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, val loss3: {val_loss3:.4f}, val acc1: {val_acc1:.4f}, val acc2: {val_acc2:.4f}'])
        if train_acc1 > 0.8 and train_acc2 > 0.3:
            train_confusion_matrix = compute_confusion_matrix(model, train_loader)
            logger.record([f'Epoch: {epoch} train confusion matrix1:\n{pd.DataFrame(train_confusion_matrix[0])}'], print_flag=False)
            logger.record([f'Epoch: {epoch} train confusion matrix2:\n{pd.DataFrame(train_confusion_matrix[1])}'], print_flag=False)
            val_confusion_matrix = compute_confusion_matrix(model, val_loader)
            logger.record([f'Epoch: {epoch} val confusion matrix1:\n{pd.DataFrame(val_confusion_matrix[0])}'], print_flag=False)
            logger.record([f'Epoch: {epoch} val confusion matrix2:\n{pd.DataFrame(val_confusion_matrix[1])}'], print_flag=False)
        save_flag = False
        if val_loss1 < min_val_loss1:
            min_val_loss1 = val_loss1
            invalid_num_epochs = 0
            save_flag = True
        else:
            invalid_num_epochs = invalid_num_epochs + 1
        if val_loss2 < min_val_loss2:
            min_val_loss2 = val_loss2
        if val_loss3 < min_val_loss3:
            min_val_loss3 = val_loss3
            invalid_num_epochs = 0
        else:
            invalid_num_epochs = invalid_num_epochs + 1
            save_flag = True
        if save_flag and checkpoint_save_path != '':
            os.makedirs(checkpoint_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))
        if max_invalid_num_epochs != -1 and invalid_num_epochs >= max_invalid_num_epochs * 2:
            break
        
        # val_loss, val_acc = evaluate(model, val_loader, criterion)
        # logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}'])
        
        # val_cm_identity, val_cm_activity = compute_confusion_matrix(model, val_loader)
        # logger.record([f'Val identity confusion matrix:\n{str(val_cm_identity)}', f'Val activity confusion matrix:\n{str(val_cm_activity)}'])

        # torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))