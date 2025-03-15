"""Loss functions and data mixing techniques."""

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Applies mixup augmentation to the data.
    
    Args:
        x: Input data (batch of images)
        y: Target labels
        alpha: Mixup alpha parameter (controls strength of interpolation)
        device: Device to use
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Applies mixup criterion to the predictions.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of targets
        y_b: Second set of targets
        lam: Mixup lambda value
    
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """Applies cutmix augmentation to the data.
    
    Args:
        x: Input data (batch of images)
        y: Target labels
        alpha: CutMix alpha parameter
        device: Device to use
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Random box coordinates
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to be the ratio of box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box coordinates for CutMix.
    
    Args:
        size: Size of the image
        lam: CutMix lambda parameter
    
    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
