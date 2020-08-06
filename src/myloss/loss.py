import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction="mean"):
    """
    https://github.com/OceanPang/Libra_R-CNN/blob/5d6096f39b90eeafaf3457f5a39572fe5e991808/mmdet/models/losses/balanced_l1_loss.py
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    assert reduction in {"none", "sum", "mean"}

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,
    )

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    https://github.com/OceanPang/Libra_R-CNN/blob/5d6096f39b90eeafaf3457f5a39572fe5e991808/mmdet/models/losses/balanced_l1_loss.py
    """

    def __init__(
        self, alpha=0.5, gamma=1.5, beta=1.0, reduction="mean", loss_weight=1.0
    ):
        super(BalancedL1Loss, self).__init__()
        assert reduction in {"none", "sum", "mean"}
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss_bbox


class FocalLoss(nn.Module):
    """
    Reference:
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-loss_bce)
        loss_f = self.alpha * (torch.tensor(1.0) - pt) ** self.gamma * loss_bce
        return loss_f.mean()


def dice(preds, targets):
    smooth = 1e-7
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()  # .float()
    union = preds_flat.sum() + targets_flat.sum()  # .float()

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return dice_score


class DiceLoss(nn.Module):
    def __init__(self, softmax=False):
        super(DiceLoss, self).__init__()
        self.softmax = softmax

    def forward(self, logits, targets):
        if self.softmax:
            # softmax channel-wise
            preds = torch.softmax(logits, dim=1)
        else:
            preds = torch.sigmoid(logits)
        return 1.0 - dice(preds, targets)


class BCEDiceLoss(nn.Module):
    """Loss defined as alpha * BCELoss - (1 - alpha) * DiceLoss"""

    def __init__(self, alpha=0.5):
        # TODO check best alpha
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        loss = self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss
        return loss


class BCEDiceLossWithRegLoss(nn.Module):
    def __init__(self, r_reg=0.5):
        super(BCEDiceLossWithRegLoss, self).__init__()
        self.r_reg = r_reg
        self.r_seg = 1.0 - r_reg

        self.reg_loss = nn.BCEWithLogitsLoss()
        self.seg_loss = BCEDiceLoss()

    def forward(self, logits_seg, logits_reg, targets_seg, targets_reg):
        loss_r = self.reg_loss(logits_reg, targets_reg)
        loss_s = self.seg_loss(logits_seg, targets_seg)
        loss = self.r_reg * loss_r + self.r_seg * loss_s
        return loss


def quadratic_kappa_coefficient(output, target):
    """
    https://www.kaggle.com/mawanda/qwk-metric-in-pytorch
    """
    output, target = output.type(torch.float32), target.type(torch.float32)
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (
        n_classes - 1
    )
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum()  # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK


def quadratic_kappa_loss(output, target, scale=2.0):
    """
    https://www.kaggle.com/mawanda/qwk-metric-in-pytorch
    """
    QWK = quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss


class QWKLoss(torch.nn.Module):
    """
    https://www.kaggle.com/mawanda/qwk-metric-in-pytorch
    """

    def __init__(self, scale=2.0, binned=False):
        super().__init__()
        self.binned = binned
        self.scale = scale

    def forward(self, output, target):
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device)
        if self.binned:
            output = torch.sigmoid(output).sum(1).round().long()
            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device)
        else:
            output = torch.softmax(output, dim=1)
        return quadratic_kappa_loss(output, target, self.scale)


def test():
    target = torch.zeros(3)
    pred = torch.rand(3)
    print(f"pred: {pred}")
    print(f"target: {target}")

    loss = BalancedL1Loss(reduction="none")
    l = loss(pred, target)
    print(l)


if __name__ == "__main__":
    test()
