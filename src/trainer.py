import random
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from dataset import TrainDataset
from factory import get_loss, get_model, get_optimizer, get_transform
from utils import (
    cutmix_tile,
    mixup_criterion,
    mixup_data_same_provider,
    quadratic_weighted_kappa,
)


def worker_init_fn(worker_id):
    random.seed(worker_id)


class LightningModuleReg(pl.LightningModule):
    """
    Lightning Module For regression
    """

    def __init__(self, cfg):
        super(LightningModuleReg, self).__init__()
        self.cfg = cfg
        self.frozen_bn = cfg.General.frozen_bn
        self.net = self.get_net()
        self.loss = get_loss(cfg.Loss)

        # Params from config
        self.mixup = self.cfg.Data.dataset.mixup
        self.cutmix = self.cfg.Data.dataset.cutmix
        self.cutmix_epoch = self.cfg.Data.dataset.cutmix_epoch
        self.out_ch = self.cfg.Model.out_channel

        # Insert after
        self.img_size = None
        self.num_tile = None
        self.tile_size = None
        self.data_provider = None
        self.softmax = None
        assert not (self.mixup and self.cutmix), "Both Mixup and Cutmix option True"

    def get_net(self):
        return get_model(self.cfg.Model)

    def on_epoch_start(self):
        if self.frozen_bn:
            print(f"frozen bn")
            self.net.eval()

    def forward(self, x, data_provider=None):
        if data_provider is None:
            return self.net(x)
        return self.net(x, data_provider)

    def _training_step_mixup(self, imgs, targets, data_provider):
        imgs, targets_a, targets_b, lam = mixup_data_same_provider(
            imgs, targets, data_provider
        )
        logits = self.forward(imgs)
        loss = mixup_criterion(self.loss, logits, targets_a, targets_b, lam)
        return loss

    def _training_step_cutmix(self, imgs, targets):
        # No cutmix end of epochs
        if self.current_epoch > self.cutmix_epoch or np.random.rand() < 0.5:
            return self._training_step_normal(imgs, targets)

        imgs, targets_a, targets_b, lam = cutmix_tile(
            imgs, targets, self.img_size, self.tile_size, beta=1.0
        )
        logits = self.forward(imgs)
        loss = mixup_criterion(self.loss, logits, targets_a, targets_b, lam)
        return loss

    def _training_step_normal(self, imgs, targets):
        logits = self.forward(imgs)
        if self.out_ch == 1:
            logits = logits.view(-1)
        loss = self.loss(logits, targets)
        return loss

    def training_step(self, batch, batch_nb):
        imgs, targets, data_provider = batch
        if self.mixup:
            loss = self._training_step_mixup(imgs, targets, data_provider)
        elif self.cutmix:
            loss = self._training_step_cutmix(imgs, targets)
        else:
            loss = self._training_step_normal(imgs, targets)
        return {"loss": loss}

    def extract_pred_target(self, logits, targets):
        if self.softmax:
            preds = logits.argmax(dim=1).detach()
            targets = targets
            return preds.detach(), targets.detach()

        if self.out_ch == 1:
            preds = logits
            targets = targets
        elif self.out_ch == 5:
            preds = logits.sigmoid().sum(1)
            targets = targets.sum(1)
        elif self.out_ch > 5:
            preds = logits[:, :5].sigmoid().sum(1)
            targets = targets[:, :5].sum(1)
        return preds.detach().round(), targets.detach().round()

    def validation_step(self, batch, batch_nb):
        imgs, targets, data_provider = batch
        logits = self.net(imgs)

        if self.out_ch == 1:
            logits = logits.view(-1)

        loss_val = self.loss(logits, targets)
        output = OrderedDict({"loss": loss_val})

        if self.out_ch > 5 and not self.softmax:
            # Loss about only label
            loss_val_5 = self.loss(logits[:, :5], targets[:, :5])
            output["loss_5"] = loss_val_5

        output["pred"], output["label"] = self.extract_pred_target(logits, targets)
        return output

    def validation_epoch_end(self, outputs):
        d = dict()

        # Validation loss
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean()
        if self.out_ch > 5 and not self.softmax:
            v_loss_5 = torch.stack([o["loss_5"] for o in outputs]).mean()
            d["v_loss_5"] = v_loss_5

        # Accuracy and Kappa score
        all_preds = torch.cat([o["pred"] for o in outputs]).cpu().numpy()
        all_targets = torch.cat([o["label"] for o in outputs]).cpu().numpy()

        # All score
        d["v_acc"] = (all_preds == all_targets).mean() * 100.0
        d["v_kappa"] = quadratic_weighted_kappa(all_targets, all_preds)

        # Score by dataprovider
        is_karolinska = self.data_provider[: len(all_preds)]
        p_k = all_preds[is_karolinska == 1]
        t_k = all_targets[is_karolinska == 1]
        p_r = all_preds[is_karolinska == 0]
        t_r = all_targets[is_karolinska == 0]
        d["v_kappa_k"] = quadratic_weighted_kappa(t_k, p_k)
        d["v_kappa_r"] = quadratic_weighted_kappa(t_r, p_r)

        ret_dict = {"progress_bar": d, "log": d.copy(), "val_loss": d["v_loss"]}
        return ret_dict

    def configure_optimizers(self):
        optimizer_cls, scheduler_cls = get_optimizer(self.cfg)

        conf_optim = self.cfg.Optimizer
        optimizer = optimizer_cls(self.parameters(), **conf_optim.optimizer.params)
        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler_default = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=10,
                total_epoch=1,
                after_scheduler=scheduler_default,
            )
        return [optimizer], [scheduler]

    def get_ds(self, phase):
        assert phase in {"train", "valid"}
        transform = get_transform(conf_augmentation=self.cfg.Augmentation[phase])
        return TrainDataset(
            conf_dataset=self.cfg.Data.dataset,
            phase=phase,
            out_ch=self.cfg.Model.out_channel,
            transform=transform,
        )

    def get_loader(self, phase):
        assert phase in {"train", "valid"}
        dataset = self.get_ds(phase=phase)

        self.softmax = dataset.softmax

        # Used for Cutmix
        self.img_size = dataset.img_size
        self.num_tile = dataset.num_tile
        self.tile_size = dataset.img_size // int(self.num_tile ** 0.5)
        if phase == "valid":
            self.data_provider = dataset.data_provider.astype(np.int)
        cfg_dataloader = self.cfg.Data.dataloader
        return DataLoader(
            dataset,
            batch_size=cfg_dataloader.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=cfg_dataloader.num_workers,
            drop_last=True if phase == "train" else False,
            worker_init_fn=worker_init_fn,
        )

    @pl.data_loader
    def train_dataloader(self):
        return self.get_loader(phase="train")

    @pl.data_loader
    def val_dataloader(self):
        return self.get_loader(phase="valid")
