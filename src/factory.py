import albumentations as alb
import albumentations.pytorch as albp
import torch
import torch.nn as nn
import torch_optimizer as toptim
import yaml
from addict import Dict

import myloss
from models import (
    CustomEfficientNet,
    CustomResnest,
    CustomResNet,
    CustomSEResNeXt,
    EUNet,
    EUNetMini,
    EUNetMini2,
    EUNetMini3,
)


"""
Functions for converting config to each object
"""


def get_transform(conf_augmentation):
    # For multi-channel mask
    additional_targets = {
        "mask0": "mask",
        "mask1": "mask",
        "mask2": "mask",
        "mask3": "mask",
        "mask4": "mask",
    }

    def get_object(trans):
        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(alb, trans.name)(augs_tmp, **trans.params)

        if hasattr(alb, trans.name):
            return getattr(alb, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]
    augs.append(albp.ToTensorV2())
    return alb.Compose(augs, additional_targets=additional_targets)


def get_model(cfg_model):
    net = None
    if "seresnext" in cfg_model.base:
        net = CustomSEResNeXt(
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pool_type=cfg_model.pool_type,
            pretrained=cfg_model.pretrained,
        )
    elif "resnest" in cfg_model.base:
        net = CustomResnest(
            base=cfg_model.base,
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pool_type=cfg_model.pool_type,
            pretrained=cfg_model.pretrained,
        )
    elif "resnet" in cfg_model.base:
        net = CustomResNet(
            base="resnet34",
            target_size=cfg_model.out_channel,
            in_ch=cfg_model.in_channel,
            pretrained=cfg_model.pretrained,
        )
    elif "eunet" == cfg_model.base:
        net = EUNet(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
        )
    elif "eunet-mini" == cfg_model.base:
        net = EUNetMini(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "eunet-mini2" == cfg_model.base:
        net = EUNetMini2(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "eunet-mini3" == cfg_model.base:
        net = EUNetMini3(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "efficientnet" in cfg_model.base:
        net = CustomEfficientNet(
            base=cfg_model.base,
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pretrained=cfg_model.pretrained,
        )
    assert net is not None
    return net


def get_loss(conf):
    conf_loss = conf.base_loss
    assert hasattr(nn, conf_loss.name) or hasattr(myloss, conf_loss.name)
    loss = None
    if hasattr(nn, conf_loss.name):
        loss = getattr(nn, conf_loss.name)
    elif hasattr(myloss, conf_loss.name):
        loss = getattr(myloss, conf_loss.name)

    if len(conf_loss.weight) > 0:
        weight = torch.Tensor(conf_loss.weight)
        conf_loss["weight"] = weight
        print(f"loss weight: {weight}")
    return loss(**conf_loss.params)


def get_optimizer(conf):
    conf_optim = conf.Optimizer
    name = conf_optim.optimizer.name
    if hasattr(torch.optim, name):
        optimizer_cls = getattr(torch.optim, name)
    else:
        optimizer_cls = getattr(toptim, name)

    if hasattr(conf_optim, "lr_scheduler"):
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim.lr_scheduler.name)
    else:
        scheduler_cls = None
    return optimizer_cls, scheduler_cls


def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


if __name__ == "__main__":
    d = read_yaml()
    print(type(d))
    print(d.Augmentation)
    print(d["Augmentation"])
    loss = get_loss(d.Loss)
    print(loss)
