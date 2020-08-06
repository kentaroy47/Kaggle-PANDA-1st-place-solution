import numpy as np
import torch


def mixup_data_same_provider(x, y, data_provider, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # 1: karolinska, 0: radboud
    indx_0 = data_provider == 0
    indx_1 = data_provider == 1

    x_0 = x[indx_0]
    x_1 = x[indx_1]
    y_0 = y[indx_0]
    y_1 = y[indx_1]

    if x_0.size(0) == 0 or x_1.size(0) == 0:
        # if only 1 data provider in batch
        return mixup_data(x, y, alpha, use_cuda=use_cuda)

    rand_index_0 = torch.randperm(x_0.size(0))
    rand_index_1 = torch.randperm(x_1.size(0))
    if use_cuda:
        rand_index_0 = rand_index_0.cuda()
        rand_index_1 = rand_index_1.cuda()

    mixed_x_0 = lam * x_0 + (1 - lam) * x_0[rand_index_0, :]
    mixed_x_1 = lam * x_1 + (1 - lam) * x_1[rand_index_1, :]
    y_a_0, y_b_0 = y_0, y_0[rand_index_0]
    y_a_1, y_b_1 = y_1, y_1[rand_index_1]

    mixed_x = torch.cat([mixed_x_0, mixed_x_1], dim=0)
    y_a = torch.cat([y_a_0, y_a_1], dim=0)
    y_b = torch.cat([y_b_0, y_b_1], dim=0)
    return mixed_x, y_a, y_b, lam


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox_tile(img_size, tile_size, lam):
    n = img_size // tile_size
    w = n
    h = n
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w) * tile_size
    bby1 = np.clip(cy - cut_h // 2, 0, h) * tile_size
    bbx2 = np.clip(cx + cut_w // 2, 0, w) * tile_size
    bby2 = np.clip(cy + cut_h // 2, 0, h) * tile_size

    return bbx1, bby1, bbx2, bby2


def cutmix_tile(x, y, img_size, tile_size, beta=1.0):
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size(0)).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox_tile(img_size, tile_size, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, target_a, target_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
