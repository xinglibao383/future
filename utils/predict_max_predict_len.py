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


def val_loss_mpjpe(model, checkpoint_filepath, device, dataloader, logger, noise_steps=None, noise_std=0.0):
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_filepath))
    model = model.to(device)
    model.eval()
    metric = Accumulator(22)
    data = None
    with torch.no_grad():
        for i, (x1, y1, z1, x2, y2, z2) in enumerate(dataloader):
            result = []
            x1, x2 = normalize(x1), normalize(x2)
            x1, y1, z1, x2, y2, z2 = x1.to(device), y1.to(device), z1.to(device), x2.to(device), y2.to(device), z2.to(device)
            y1_hat, x2_hat, y2_hat = model(x1)
            y1_hat, y1 = y1_hat.clamp(min=-0.9999, max=0.9999), y1.clamp(min=-0.9999, max=0.9999)
            y2_hat, y2 = y2_hat.clamp(min=-0.9999, max=0.9999), y2.clamp(min=-0.9999, max=0.9999)
            y1_hat, y1 = torch.atanh(y1_hat), torch.atanh(y1)
            y2_hat, y2 = torch.atanh(y2_hat), torch.atanh(y2)
            y1_hat, y1 = y1_hat * z1, y1 * z1
            y2_hat, y2 = y2_hat * z2[:, :15, :, :], y2 * z2
            error1, error2 = torch.norm(y1_hat - y1, dim=-1).mean(), torch.norm(y2_hat - y2[:, :15, :, :], dim=-1).mean()
            result.extend([error1.sum().item(), error1.numel(), error2.sum().item(), error2.numel()])
            for j in range(9):
                x2_hat = x2_hat.detach()
                # ========================= 噪声注入 =========================
                if noise_steps is not None and j in noise_steps:
                    std = x2_hat.std(dim=(0, 2), keepdim=True)
                    x2_hat = x2_hat + torch.randn_like(x2_hat) * noise_std * std
                # ============================================================
                x1 = torch.cat([x1[:, :, 50:], x2_hat], dim=-1)
                _, x2_hat, y2_hat = model(x1)
                y2_hat = y2_hat.clamp(min=-0.9999, max=0.9999)
                y2_hat = torch.atanh(y2_hat)
                y2_hat = y2_hat * z2[:, (j + 1) * 15:(j + 2) * 15, :, :]
                error = torch.norm(y2_hat - y2[:, (j + 1) * 15:(j + 2) * 15, :, :], dim=-1).mean()
                result.extend([error.sum().item(), error.numel()])
            metric.add(*result)
            # logger.record([", ".join([f"val mpjpe{i}: {v:.4f}" for i, v in enumerate([metric[i] / metric[i + 1] for i in range(0, 22, 2)])])])
            data = [metric[i] / metric[i + 1] for i in range(0, 22, 2)]
            logger.record([", ".join([f"val mpjpe{i}: {v:.4f}" for i, v in enumerate(data)])])
    return data