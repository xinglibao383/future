import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(model, dataloader):
    device = next(iter(model.parameters())).device
    all_preds_y1, all_labels_y1 = [], []
    all_preds_y2, all_labels_y2 = [], []

    model.eval()
    with torch.no_grad():
        for x1, y1, _, y2 in dataloader:
            x1, y1, y2 = x1.to(device), y1.to(device), y2.to(device)
            y1_hat, _, y2_hat = model(x1)

            all_preds_y1.extend(torch.argmax(y1_hat, dim=1).cpu().numpy())
            all_labels_y1.extend(y1.cpu().numpy())

            all_preds_y2.extend(torch.argmax(y2_hat, dim=1).cpu().numpy())
            all_labels_y2.extend(y2.cpu().numpy())

    cm_y1 = confusion_matrix(all_labels_y1, all_preds_y1)
    cm_y2 = confusion_matrix(all_labels_y2, all_preds_y2)

    normalized_cm_y1 = cm_y1.astype('float') / cm_y1.sum(axis=1, keepdims=True)
    normalized_cm_y2 = cm_y2.astype('float') / cm_y2.sum(axis=1, keepdims=True)

    return np.round(normalized_cm_y1, 2), np.round(normalized_cm_y2, 2)