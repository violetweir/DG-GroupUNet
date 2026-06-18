import torch
from torch import nn

__all__ = ["CMUNeXt", "cmunext", "cmunext_s", "cmunext_l"]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(ch_in, ch_in, kernel_size=k, groups=ch_in, padding=k // 2),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        return self.up(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXt(nn.Module):
    def __init__(
        self,
        input_channel=3,
        in_channels=None,
        num_classes=1,
        dims=None,
        depths=None,
        kernels=None,
    ):
        super().__init__()
        if dims is None:
            dims = [16, 32, 128, 160, 256]
        if depths is None:
            depths = [1, 1, 1, 3, 1]
        if kernels is None:
            kernels = [3, 3, 7, 7, 7]
        if in_channels is not None:
            input_channel = in_channels

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        return self.Conv_1x1(d2)


def cmunext(dims=None, depths=None, kernels=None, **kwargs):
    if dims is None:
        dims = [16, 32, 128, 160, 256]
    if depths is None:
        depths = [1, 1, 1, 3, 1]
    if kernels is None:
        kernels = [3, 3, 7, 7, 7]
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, **kwargs)


def cmunext_s(dims=None, depths=None, kernels=None, **kwargs):
    if dims is None:
        dims = [8, 16, 32, 64, 128]
    if depths is None:
        depths = [1, 1, 1, 1, 1]
    if kernels is None:
        kernels = [3, 3, 7, 7, 9]
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, **kwargs)


def cmunext_l(dims=None, depths=None, kernels=None, **kwargs):
    if dims is None:
        dims = [32, 64, 128, 256, 512]
    if depths is None:
        depths = [1, 1, 1, 6, 3]
    if kernels is None:
        kernels = [3, 3, 7, 7, 7]
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, **kwargs)
