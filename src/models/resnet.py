import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


"""
Sample Model(ResNet18)
"""


class CustomResNet(nn.Module):
    def __init__(self, base="resnet18", target_size=1, in_ch=3, pretrained=False):
        super().__init__()
        assert base in {"resnet18", "resnet34", "resnet50"}
        assert in_ch % 3 == 0
        self.base = base
        self.in_ch = in_ch

        if base == "resnet18":
            self.model = resnet18(pretrained=pretrained)
        elif base == "resnet34":
            self.model = resnet34(pretrained=pretrained)
        elif base == "resnet50":
            self.model = resnet50(pretrained=pretrained)

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(self.model.fc.in_features, target_size)

        if in_ch != 3:
            old_in_ch = 3
            old_conv = self.model.conv1

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

            self.model.conv1 = new_conv
            self.model.conv1.weight = nn.Parameter(new_weight)

    def forward(self, x):
        x = self.model(x)
        return x


def test():
    from torchsummary import summary

    model = CustomResNet(base="resnet50", pretrained=False)
    print(model)

    summary(model, (3, 1024, 1024))


if __name__ == "__main__":
    test()
