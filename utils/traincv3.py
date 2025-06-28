import os
import torch
import numpy as np
from torch import nn
from utils.accumulator import Accumulator
from utils.dataloader import *
from sklearn.metrics import confusion_matrix


def accuracy(y_hat, y):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def compute_confusion_matrix(model, dataloader):
    device = next(iter(model.parameters())).device
    all_identity_preds, all_identity_labels = [], []
    all_activity_preds, all_activity_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y1, y2 in dataloader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            y1_hat, y2_hat = model(x)

            all_identity_preds.extend(torch.argmax(y1_hat, dim=1).view(-1).cpu().numpy())
            all_identity_labels.extend(y1.view(-1).cpu().numpy())

            all_activity_preds.extend(torch.argmax(y2_hat, dim=1).view(-1).cpu().numpy())
            all_activity_labels.extend(y2.view(-1).cpu().numpy())
            

    cm_identity = confusion_matrix(all_identity_labels, all_identity_preds)
    cm_activity = confusion_matrix(all_activity_labels, all_activity_preds)

    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_cm_identity = np.nan_to_num(cm_identity.astype('float') / cm_identity.sum(axis=1, keepdims=True))
        normalized_cm_activity = np.nan_to_num(cm_activity.astype('float') / cm_activity.sum(axis=1, keepdims=True))

    return np.round(normalized_cm_identity, 2), np.round(normalized_cm_activity, 2)


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


def train(model, train_loader, val_loader, lr, weight_decay, mask_ratio, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    max_acc1 = -1
    max_acc2 = -1

    alpha, beta, gamma = 1.0, 1.0, 1.0

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

            alpha = 1.0 / max(acc2, 1e-5) * float(loss3.item())
            beta = 1.0 / max(loss2.item(), 1e-5)
            gamma = 1.0 / max(acc1, 1e-5) * float(loss1.item())

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
        print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train acc1: {train_acc1:.4f}, train acc2: {train_acc2:.4f}')
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, train loss1: {train_loss1:.4f}, train loss2: {train_loss2:.4f}, train loss3: {train_loss3:.4f}, train acc1: {train_acc1:.4f}, train acc2: {train_acc2:.4f}'])
        
        val_loss, val_loss1, val_loss2, val_loss3, val_acc1, val_acc2 = evaluate(model, val_loader, criterion1, criterion2)
        print(f'Epoch: {epoch}, val loss: {val_loss:.4f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, val loss3: {val_loss3:.4f}, val acc1: {val_acc1:.4f}, val acc2: {val_acc2:.4f}')
        logger.record([f'Epoch: {epoch}, val loss: {val_loss:.4f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, val loss3: {val_loss3:.4f}, val acc1: {val_acc1:.4f}, val acc2: {val_acc2:.4f}'])
        
        if val_acc1 >= max_acc1 or val_acc2 >= max_acc2:
            max_acc1 = val_acc1
            max_acc2 = val_acc2
            # torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))
        
        # val_loss, val_acc = evaluate(model, val_loader, criterion)
        # logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}'])
        
        # val_cm_identity, val_cm_activity = compute_confusion_matrix(model, val_loader)
        # logger.record([f'Val identity confusion matrix:\n{str(val_cm_identity)}', f'Val activity confusion matrix:\n{str(val_cm_activity)}'])

        # torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))