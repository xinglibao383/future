import torch
from collections import Counter


def compute_class_weights(label_list, num_classes=34):
    counts = Counter(label_list)
    total = sum(counts.values())
    weights = [0.0] * num_classes
    for label, count in counts.items():
        weights[label] = total / count
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    return weights


def get_class_weights(train_loader, val_loader, logger, num_classes=34):
    y1_train = torch.cat([y1 for _, y1, _, _ in train_loader]).tolist()
    y2_train = torch.cat([y2 for _, _, _, y2 in train_loader]).tolist()
    y1_val = torch.cat([y1 for _, y1, _, _ in val_loader]).tolist()
    y2_val = torch.cat([y2 for _, _, _, y2 in val_loader]).tolist()

    logger.record([f"Train y1 count: {Counter(y1_train)}"])
    logger.record([f"Train y2 count: {Counter(y2_train)}"])
    logger.record([f"Val y1 count: {Counter(y1_val)}"])
    logger.record([f"Val y2 count: {Counter(y2_val)}"])

    y1_weights = compute_class_weights(y1_train, num_classes)
    y2_weights = compute_class_weights(y2_train, num_classes)
    y1_y2_weights = compute_class_weights(y1_train + y2_train, num_classes)

    logger.record([f"y1 class weights: {y1_weights}"])
    logger.record([f"y2 class weights: {y2_weights}"])
    logger.record([f"y1 + y2 class weights: {y1_y2_weights}"])

    return y1_weights, y2_weights, y1_y2_weights

