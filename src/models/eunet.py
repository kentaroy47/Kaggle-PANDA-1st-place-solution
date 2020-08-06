import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from .layer import AdaptiveConcatPool2d, Decoder, Flatten, GeM, SEBlock


class Test(nn.Module):
    def __init__(self, base="efficientnet-b0"):
        super(Test, self).__init__()
        self.base = base
        self.net = EfficientNet.from_name(base)

    def forward(self, x):
        outputs = list()

        # Stem
        x = self.net._swish(self.net._bn0(self.net._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self.net._blocks):
            drop_connect_rate = self.net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.net._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            print(f"idx: {idx}, x.shape: {x.size()}")

        # Head
        x = self.net._swish(self.net._bn1(self.net._conv_head(x)))

        return x, outputs


class ENetBackbone(nn.Module):
    def __init__(
        self, base="efficientnet-b0", pretrained=False,
    ):
        super(ENetBackbone, self).__init__()
        assert base in {
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
        }

        self.base = base
        self.pretrained = pretrained

        if pretrained:
            self.net = EfficientNet.from_pretrained(base)
        else:
            self.net = EfficientNet.from_name(base)

        # Segmentation info
        # four dcoder
        model_name_to_extracts = {
            "efficientnet-b0": {2, 4, 10, 15},
            "efficientnet-b1": {4, 7, 15, 22},
            "efficientnet-b2": {4, 7, 15, 22},
            "efficientnet-b3": {4, 7, 14, 25},
            "efficientnet-b4": {5, 9, 21, 31},
        }
        self.extracts = model_name_to_extracts[base]
        self.len_encoder = max(self.extracts) + 1

        # four decoder
        model_name_to_planes = {
            "efficientnet-b0": [24, 40, 112, 320],
            "efficientnet-b1": [24, 40, 112, 320],
            "efficientnet-b2": [24, 48, 120, 352],
            "efficientnet-b3": [32, 48, 136, 384],
            "efficientnet-b4": [32, 56, 160, 448],
        }
        self.planes = model_name_to_planes[base]

    def forward(self, x):
        outputs = list()

        # Stem
        x = self.net._swish(self.net._bn0(self.net._conv_stem(x)))

        # Blocks
        # for idx in range(self.len_encoder):
        for idx in range(len(self.net._blocks)):
            drop_connect_rate = self.net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.net._blocks)
            x = self.net._blocks[idx](x, drop_connect_rate=drop_connect_rate)

            if idx in self.extracts:
                outputs.append(x)

        # Head
        x = self.net._swish(self.net._bn1(self.net._conv_head(x)))
        return x, outputs


class ConvBnRelu2d(nn.Module):
    def __init__(
        self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
    ):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class ClsHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ClsHead, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cls_head = nn.Sequential(
            GeM(), Flatten(), SEBlock(in_ch), nn.Dropout(), nn.Linear(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.cls_head(x)
        return x


class EUNetMini(nn.Module):
    """
    split branch by data_provider
    """

    def __init__(self, base, cls_out_ch, seg_out_ch, pretrained=False, fp16=False):
        super(EUNetMini, self).__init__()
        self.encoder = ENetBackbone(base=base, pretrained=pretrained)
        self.planes = self.encoder.planes
        self.fp16 = fp16

        self.center = nn.Sequential(
            ConvBnRelu2d(self.planes[3], self.planes[3], kernel_size=3, padding=1),
            ConvBnRelu2d(self.planes[3], self.planes[2], kernel_size=3, padding=1),
        )

        base_ch = 32
        kwargs_decoder = {
            "attention_type": "cbam",
            "attention_kernel_size": 1,
            "reduction": 16,
            "out_channels": 32,
        }

        self.decoder4 = Decoder(self.planes[3] + self.planes[2], 512, **kwargs_decoder)
        self.decoder3 = Decoder(self.planes[2] + base_ch, 256, **kwargs_decoder)
        self.decoder2 = Decoder(self.planes[1] + base_ch, 128, **kwargs_decoder)
        self.decoder1 = Decoder(self.planes[0] + base_ch, 64, **kwargs_decoder)

        self.final_1 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        self.final_2 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        self.cls_head = ClsHead(
            in_ch=self.encoder.net._fc.in_features, out_ch=cls_out_ch
        )

    def forward(self, x, is_karolinska):
        x, outputs = self.encoder(x)
        e2, e3, e4, e5 = outputs  # 1/4, 1/8, 1/16, 1/32

        c = self.center(e5)

        d5 = self.decoder4(torch.cat([c, e5], 1))  # 1/16
        d4 = self.decoder3(torch.cat([d5, e4], 1))  # 1/8
        d3 = self.decoder2(torch.cat([d4, e3], 1))  # 1/4
        d2 = self.decoder1(torch.cat([d3, e2], 1))  # 1/2

        f = torch.cat(
            (
                d2,
                F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False),
                F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=False),
                F.interpolate(d5, scale_factor=8, mode="bilinear", align_corners=False),
            ),
            1,
        )

        f1 = self.final_1(f)
        f2 = self.final_2(f)
        is_karolinska = is_karolinska.view(-1, 1, 1, 1)
        if self.fp16:
            is_karolinska = is_karolinska.half()
        seg_logit = f1 * is_karolinska + f2 * (1.0 - is_karolinska)

        # Classification
        cls_logit = self.cls_head(x)
        return seg_logit, cls_logit

    def pred_cls(self, x, is_karolinska):
        x, _ = self.encoder(x)  # 1/4, 1/8, 1/16, 1/32
        cls_logit = self.cls_head(x)
        return cls_logit


class EUNetMini2(nn.Module):
    def __init__(self, base, cls_out_ch, seg_out_ch, pretrained=False, fp16=False):
        super(EUNetMini2, self).__init__()
        self.encoder = ENetBackbone(base=base, pretrained=pretrained)
        self.planes = self.encoder.planes
        self.fp16 = fp16

        self.center = nn.Sequential(
            ConvBnRelu2d(self.planes[3], self.planes[3], kernel_size=3, padding=1),
            ConvBnRelu2d(self.planes[3], self.planes[2], kernel_size=3, padding=1),
        )

        base_ch = 32
        kwargs_decoder = {
            "attention_type": "cbam",
            "attention_kernel_size": 1,
            "reduction": 16,
            "out_channels": 32,
        }

        self.decoder4 = Decoder(self.planes[3] + self.planes[2], 512, **kwargs_decoder)
        self.decoder3 = Decoder(self.planes[2] + base_ch, 256, **kwargs_decoder)
        self.decoder2 = Decoder(self.planes[1] + base_ch, 128, **kwargs_decoder)
        self.decoder1 = Decoder(self.planes[0] + base_ch, 64, **kwargs_decoder)

        self.final_1 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        self.final_2 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        # Make mask feature
        mask_feature_size = 48  # input_size / 32
        self.pool = nn.AdaptiveMaxPool2d((mask_feature_size, mask_feature_size))
        base_ch2 = 96
        self.mask_score_fcn1 = nn.Conv2d(base_ch * 4, base_ch2 * 2, 3, 1, 1)
        self.mask_score_fcn2 = nn.Conv2d(base_ch2 * 2, base_ch2, 3, 1, 1)

        self.cls_head = ClsHead(
            in_ch=self.encoder.net._fc.in_features + base_ch2, out_ch=cls_out_ch,
        )

    def forward(self, x, is_karolinska):
        x, outputs = self.encoder(x)
        e2, e3, e4, e5 = outputs  # 1/4, 1/8, 1/16, 1/32

        c = self.center(e5)

        d5 = self.decoder4(torch.cat([c, e5], 1))  # 1/16
        d4 = self.decoder3(torch.cat([d5, e4], 1))  # 1/8
        d3 = self.decoder2(torch.cat([d4, e3], 1))  # 1/4
        d2 = self.decoder1(torch.cat([d3, e2], 1))  # 1/2

        f = torch.cat(
            (
                d2,
                F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False),
                F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=False),
                F.interpolate(d5, scale_factor=8, mode="bilinear", align_corners=False),
            ),
            1,
        )

        f1 = self.final_1(f)
        f2 = self.final_2(f)
        is_karolinska = is_karolinska.view(-1, 1, 1, 1)
        if self.fp16:
            is_karolinska = is_karolinska.half()
        seg_logit = f1 * is_karolinska + f2 * (1.0 - is_karolinska)

        m = self.pool(f)
        m = self.mask_score_fcn1(m)
        m = self.mask_score_fcn2(m)

        # Classification
        x = torch.cat((x, m), 1)
        cls_logit = self.cls_head(x)
        return seg_logit, cls_logit

    def pred_cls(self, x, is_karolinska):
        _, cls_logit = self.forward(x, is_karolinska)
        return cls_logit


class EUNetMini3(nn.Module):
    def __init__(self, base, cls_out_ch, seg_out_ch, pretrained=False, fp16=False):
        super(EUNetMini3, self).__init__()
        self.encoder = ENetBackbone(base=base, pretrained=pretrained)
        self.planes = self.encoder.planes
        self.fp16 = fp16

        self.center = nn.Sequential(
            ConvBnRelu2d(self.planes[3], self.planes[3], kernel_size=3, padding=1),
            ConvBnRelu2d(self.planes[3], self.planes[2], kernel_size=3, padding=1),
        )

        base_ch = 32
        kwargs_decoder = {
            "attention_type": "cbam",
            "attention_kernel_size": 1,
            "reduction": 16,
            "out_channels": 32,
        }

        self.decoder4 = Decoder(self.planes[3] + self.planes[2], 512, **kwargs_decoder)
        self.decoder3 = Decoder(self.planes[2] + base_ch, 256, **kwargs_decoder)
        self.decoder2 = Decoder(self.planes[1] + base_ch, 128, **kwargs_decoder)
        self.decoder1 = Decoder(self.planes[0] + base_ch, 64, **kwargs_decoder)

        self.final_1 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        self.final_2 = nn.Sequential(
            ConvBnRelu2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        # Make mask feature
        mask_feature_size = 96  # input_size / 32
        self.pool = AdaptiveConcatPool2d((mask_feature_size, mask_feature_size))

        base_ch2 = 128
        self.mask_score_fcn = nn.Sequential(
            ConvBnRelu2d(seg_out_ch * 4, base_ch2, kernel_size=3, stride=1, padding=1),
            ConvBnRelu2d(base_ch2, base_ch2 * 2, kernel_size=3, stride=1, padding=1),
            ConvBnRelu2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_ch2, base_ch2, 3, 2, 1),
        )

        self.cls_head = ClsHead(
            in_ch=self.encoder.net._fc.in_features + base_ch2, out_ch=cls_out_ch,
        )

    def forward(self, x, is_karolinska):
        x, outputs = self.encoder(x)
        e2, e3, e4, e5 = outputs  # 1/4, 1/8, 1/16, 1/32

        c = self.center(e5)

        d5 = self.decoder4(torch.cat([c, e5], 1))  # 1/16
        d4 = self.decoder3(torch.cat([d5, e4], 1))  # 1/8
        d3 = self.decoder2(torch.cat([d4, e3], 1))  # 1/4
        d2 = self.decoder1(torch.cat([d3, e2], 1))  # 1/2

        f = torch.cat(
            (
                d2,
                F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False),
                F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=False),
                F.interpolate(d5, scale_factor=8, mode="bilinear", align_corners=False),
            ),
            1,
        )

        f1 = self.final_1(f)
        f2 = self.final_2(f)
        is_karolinska = is_karolinska.view(-1, 1, 1, 1)
        if self.fp16:
            is_karolinska = is_karolinska.half()
        seg_logit = f1 * is_karolinska + f2 * (1.0 - is_karolinska)

        s = torch.cat((f1, f2), 1)
        m = self.pool(s)
        m = self.mask_score_fcn(m)

        # Classification
        x = torch.cat((x, m), 1)
        cls_logit = self.cls_head(x)
        return seg_logit, cls_logit

    def pred_cls(self, x, is_karolinska):
        _, cls_logit = self.forward(x, is_karolinska)
        return cls_logit


class EUNet(nn.Module):
    """Basic UNet with hyper columns"""

    def __init__(
        self, base, cls_out_ch, seg_out_ch, pretrained=False,
    ):
        super(EUNet, self).__init__()
        self.encoder = ENetBackbone(base=base, pretrained=pretrained)
        self.planes = self.encoder.planes

        self.center = nn.Sequential(
            ConvBnRelu2d(self.planes[3], self.planes[3], kernel_size=3, padding=1),
            ConvBnRelu2d(self.planes[3], self.planes[2], kernel_size=3, padding=1),
        )

        base_ch = 32
        kwargs_decoder = {
            "attention_type": "cbam",
            "attention_kernel_size": 1,
            "reduction": 16,
            "out_channels": 32,
        }
        self.decoder5 = Decoder(self.planes[3] + self.planes[2], 512, **kwargs_decoder)
        self.decoder4 = Decoder(self.planes[2] + base_ch, 256, **kwargs_decoder)
        self.decoder3 = Decoder(self.planes[1] + base_ch, 128, **kwargs_decoder)
        self.decoder2 = Decoder(self.planes[0] + base_ch, 64, **kwargs_decoder)
        self.decoder1 = Decoder(base_ch, 32, **kwargs_decoder)

        self.final = nn.Sequential(
            ConvBnRelu2d(base_ch * 5, base_ch * 2, kernel_size=3, padding=1),
            ConvBnRelu2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.Conv2d(base_ch, seg_out_ch, kernel_size=1, padding=0),
        )

        self.cls_head = ClsHead(
            in_ch=self.encoder.net._fc.in_features, out_ch=cls_out_ch
        )

    def forward(self, x):
        x, outputs = self.encoder(x)  # 1/4, 1/8, 1/16, 1/32
        e2, e3, e4, e5 = outputs

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        f = torch.cat(
            (
                d1,
                F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False),
                F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=False),
                F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=False),
                F.interpolate(
                    d5, scale_factor=16, mode="bilinear", align_corners=False
                ),
            ),
            1,
        )
        seg_logit = self.final(f)

        # Classification
        cls_logit = self.cls_head(x)
        return seg_logit, cls_logit

    def pred_cls(self, x):
        x, _ = self.encoder(x)  # 1/4, 1/8, 1/16, 1/32
        cls_logit = self.cls_head(x)
        return cls_logit


def test():
    from torchsummary import summary

    net = EUNetMini3(
        base="efficientnet-b0", cls_out_ch=1, seg_out_ch=4, pretrained=False
    )
    print(net.cls_head)
    # net = Test(base="efficientnet-b2")
    # summary(net, (3, 256, 256))
    input = torch.zeros((2, 3, 1536, 1536))
    is_karolinska = torch.Tensor([[0], [1]])

    seg_out, cls_out = net(input, is_karolinska)
    print("input: ", input.size())
    print("segout: ", seg_out.size())
    print("clsout: ", cls_out.size())
    # out = net.pred_cls(input)
    # print("pred clsout: ", out.size())


if __name__ == "__main__":
    test()
