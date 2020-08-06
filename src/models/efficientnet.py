import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from layer import AdaptiveConcatPool2d, Flatten, GeM, SEBlock


class CustomEfficientNet(nn.Module):
    def __init__(
        self,
        base="efficientnet-b0",
        pool_type="gem",
        in_ch=3,
        out_ch=1,
        pretrained=False,
    ):
        super(CustomEfficientNet, self).__init__()
        assert base in {
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
        }
        assert pool_type in {"concat", "avg", "gem"}

        self.base = base
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pretrained = pretrained

        if pretrained:
            self.net = EfficientNet.from_pretrained(base)
        else:
            self.net = EfficientNet.from_name(base)

        out_shape = self.net._fc.in_features
        if pool_type == "concat":
            self.net._avg_pooling = AdaptiveConcatPool2d()
            out_shape = out_shape * 2
        elif pool_type == "gem":
            self.net._avg_pooling = GeM()
            out_shape = out_shape
        self.net._fc = nn.Sequential(
            Flatten(), SEBlock(out_shape), nn.Dropout(), nn.Linear(out_shape, out_ch)
        )

        if in_ch != 3:
            old_in_ch = 3
            old_conv = self.net._conv_stem

            # Make new weight
            weight = old_conv.weight
            new_weight = torch.cat([weight] * (self.in_ch // old_in_ch), dim=1)

            # Make new conv
            new_conv = nn.Conv2d(
                in_channels=self.in_ch,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )

            self.net._conv_stem = new_conv
            self.net._conv_stem.weight = nn.Parameter(new_weight)

    def forward(self, x):
        x = self.net(x)
        return x


def test():
    from torchsummary import summary

    in_ch = 102

    net = CustomEfficientNet(in_ch=in_ch, base="efficientnet-b0", pool_type="gem")
    print(net)

    # summary(net, (6, 224, 224))
    input = torch.rand((2, in_ch, 128, 128))
    output = net(input)
    print(input.size())
    print(output.size())
    print(output)


if __name__ == "__main__":
    test()
