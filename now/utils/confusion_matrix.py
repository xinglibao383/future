import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(model, dataloader):
    device = next(iter(model.parameters())).device
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            all_preds.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum = cm.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    return np.round(normalized_cm, 2)