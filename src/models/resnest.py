##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""ResNet variants"""
import math

import torch
import torch.nn as nn

from .layer import AdaptiveConcatPool2d, Flatten, GeM, SEBlock
from .split_attn import SplAtConv2d

__all__ = [
    "ResNet",
    "Bottleneck",
    "CustomResnest",
    "resnest50",
    "resnest101",
    "resnest50_fast_2s2x40d",
]


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        rectified_conv=False,
        rectify_avg=False,
        norm_layer=None,
        dropblock_prob=0.0,
        last_gamma=False,
    ):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        else:
            self.conv2 = nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_

            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        rectified_conv=False,
        rectify_avg=False,
        norm_layer=None,
        dropblock_prob=0.0,
        last_gamma=False,
        reduction=16,
    ):
        super(SEBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        else:
            self.conv2 = nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_

            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.se_module = SEModule(planes * 4, reduction=reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # pylint: disable=unused-variable
    def __init__(
        self,
        block,
        layers,
        radix=1,
        groups=1,
        bottleneck_width=64,
        num_classes=1000,
        dilated=False,
        dilation=1,
        deep_stem=False,
        stem_width=64,
        avg_down=False,
        rectified_conv=False,
        rectify_avg=False,
        avd=False,
        avd_first=False,
        final_drop=0.0,
        dropblock_prob=0,
        last_gamma=False,
        norm_layer=nn.BatchNorm2d,
    ):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        conv_layer = nn.Conv2d
        conv_kwargs = {"average_mode": rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(
                    3,
                    stem_width,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                ),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                ),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                ),
            )
        else:
            self.conv1 = conv_layer(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_layer=norm_layer, is_first=False
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer
        )
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=1,
                dilation=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=1,
                dilation=4,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        elif dilation == 2:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                dilation=1,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=1,
                dilation=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        else:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_layer=None,
        dropblock_prob=0.0,
        is_first=True,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(
                        nn.AvgPool2d(
                            kernel_size=stride,
                            stride=stride,
                            ceil_mode=True,
                            count_include_pad=False,
                        )
                    )
                else:
                    down_layers.append(
                        nn.AvgPool2d(
                            kernel_size=1,
                            stride=1,
                            ceil_mode=True,
                            count_include_pad=False,
                        )
                    )
                down_layers.append(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    )
                )
            else:
                down_layers.append(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                )
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=1,
                    is_first=is_first,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )
        elif dilation == 4:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=2,
                    is_first=is_first,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=dilation,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


_url_format = "https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth"

_model_sha256 = {
    name: checksum
    for checksum, name in [
        ("528c19ca", "resnest50"),
        ("22405ba7", "resnest101"),
        ("9d126481", "resnest50_fast_2s2x40d"),
    ]
}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError(
            "Pretrained model for {name} is not available.".format(name=name)
        )
    return _model_sha256[name][:8]


resnest_model_urls = {
    name: _url_format.format(name, short_hash(name)) for name in _model_sha256.keys()
}


def resnest50(pretrained=False, root="~/.encoding/models", **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest50"], progress=True, check_hash=True
            )
        )
    return model


def se_resnest50(pretrained=False, root="~/.encoding/models", **kwargs):
    model = ResNet(
        SEBottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    if pretrained:
        # TODO: change !!!
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest50"], progress=True, check_hash=True
            ),
            strict=False,
        )
    return model


def resnest101(pretrained=False, root="~/.encoding/models", **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest101"], progress=True, check_hash=True
            )
        )
    return model


def resnest50_fast_2s2x40d(pretrained=False, root="~/.encoding/models", **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=2,
        bottleneck_width=40,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest50_fast_2s2x40d"],
                progress=True,
                check_hash=True,
            )
        )
    return model


def se_resnest50_fast_2s2x40d(pretrained=False, root="~/.encoding/models", **kwargs):
    model = ResNet(
        SEBottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=2,
        bottleneck_width=40,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    if pretrained:
        # TODO using se module weight of seresnext ???
        # Skip load SE module weight (these weights are not exists)
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest50_fast_2s2x40d"],
                progress=True,
                check_hash=True,
            ),
            strict=False,
        )
    return model


class CustomResnest(nn.Module):
    """
    Refer from https://github.com/okotaku/kaggle_rsna2019_3rd_solution/blob/master/src/model.py
    """

    def __init__(
        self, in_ch=3, out_ch=1, pool_type="concat", base="resnest50", pretrained=False,
    ):
        assert pool_type in {"concat", "avg", "gem"}
        assert in_ch % 3 == 0
        assert base in {
            "resnest50",
            "resnest101",
            "resnest50_fast_2s2x40d",
            "se_resnest50",
            "se_resnest50_fast_2s2x40d",
        }
        super().__init__()
        self.base = base

        if base == "resnest50":
            self.net = resnest50(pretrained=pretrained)
        elif base == "resnest101":
            self.net = resnest101(pretrained=pretrained)
        elif base == "resnest50_fast_2s2x40d":
            self.net = resnest50_fast_2s2x40d(pretrained=pretrained)
        elif base == "se_resnest50":
            self.net = se_resnest50(pretrained=pretrained)
        elif base == "se_resnest50_fast_2s2x40d":
            self.net = se_resnest50_fast_2s2x40d(pretrained=pretrained)

        out_shape = 2048
        if pool_type == "concat":
            self.net.avgpool = AdaptiveConcatPool2d()
            out_shape = out_shape * 2
        elif pool_type == "avg":
            self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out_shape = out_shape
        elif pool_type == "gem":
            self.net.avgpool = GeM()
            out_shape = out_shape
        self.net.fc = nn.Sequential(
            Flatten(), SEBlock(out_shape), nn.Dropout(), nn.Linear(out_shape, out_ch)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def test():
    from torchsummary import summary

    net = CustomResnest(
        pretrained=True, base="se_resnest50_fast_2s2x40d", pool_type="gem"
    )
    print(net.net.bn1.bias)  # to check load weights
    # print(net)
    # summary(net, (3, 224, 224))


if __name__ == "__main__":
    test()
