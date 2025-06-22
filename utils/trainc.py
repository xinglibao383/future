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


def evaluate(model, dataloader, criterion):
    metric = Accumulator(3)
    device = next(iter(model.parameters())).device

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            acc = accuracy(y_hat, y)

            metric.add(loss.item() * batch_size, 
                       acc * batch_size, 
                       batch_size)

    return metric[0] / metric[2], metric[1] / metric[2]


def train(model, train_loader, val_loader, lr, num_epochs, devices, checkpoint_save_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # 训练损失之和, 正确预测的样本数量, 样本数量
        metric = Accumulator(3)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            

            optimizer.zero_grad()
            batch_size = x.shape[0]
            x, y = x.to(devices[0]), y.to(devices[0])
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            acc = accuracy(y_hat, y)

            metric.add(loss.item() * batch_size, 
                       acc * batch_size,
                       batch_size)

            if i % 20 == 0:
                train_loss = metric[0] / metric[2]
                print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.4f}')
        
        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        logger.record([f'Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}'])
        
        # val_cm_identity, val_cm_activity = compute_confusion_matrix(model, val_loader)
        # logger.record([f'Val identity confusion matrix:\n{str(val_cm_identity)}', f'Val activity confusion matrix:\n{str(val_cm_activity)}'])

        # torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"checkpoint_{epoch}.pth"))