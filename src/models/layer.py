"""
https://github.com/okotaku/kaggle_rsna2019_3rd_solution/blob/master/src/layer.py
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class MaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, x.shape[2:])


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = cSE + sSE

        return x


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"
        )


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return sinusoid_table


def get_sinusoid_encoding_table_2d(H, W, d_hid):
    """ Sinusoid position encoding table """
    n_position = H * W
    sinusoid_table = get_sinusoid_encoding_table(n_position, d_hid)
    sinusoid_table = sinusoid_table.reshape(H, W, d_hid)
    return sinusoid_table


class ConvBn2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class ConvBnRelu2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class CBAM_Module(nn.Module):
    def __init__(
        self, channels, reduction=4, attention_kernel_size=3, position_encode=False
    ):
        super(CBAM_Module, self).__init__()
        self.position_encode = position_encode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        if self.position_encode:
            k = 3
        else:
            k = 2
        self.conv_after_concat = nn.Conv2d(
            k,
            1,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=attention_kernel_size // 2,
        )
        self.sigmoid_spatial = nn.Sigmoid()
        self.position_encoded = None

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        if self.position_encode:
            if self.position_encoded is None:
                pos_enc = get_sinusoid_encoding_table(h, w)
                pos_enc = Variable(torch.FloatTensor(pos_enc), requires_grad=False)
                if x.is_cuda:
                    pos_enc = pos_enc.cuda()
                self.position_encoded = pos_enc
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        if self.position_encode:
            pos_enc = self.position_encoded
            pos_enc = pos_enc.view(1, 1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat((avg, mx, pos_enc), 1)
        else:
            x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        middle_channels,
        out_channels,
        up_sample=True,
        attention_type=None,
        attention_kernel_size=3,
        reduction=16,
        reslink=False,
    ):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.up_sample = up_sample
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.reslink = reslink
        if attention_type is None:
            pass
        elif attention_type.find("cbam") >= 0:
            self.channel_gate = CBAM_Module(
                out_channels, reduction, attention_kernel_size
            )
        if self.reslink:
            self.shortcut = ConvBn2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

    def forward(self, x, size=None):
        if self.up_sample:
            if size is None:
                x = F.interpolate(
                    x, scale_factor=2, mode="bilinear", align_corners=True
                )  # False
            else:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        if self.reslink:
            shortcut = self.shortcut(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.attention:
            x = self.channel_gate(x)
        if self.reslink:
            x = F.relu(x + shortcut)
        return x
