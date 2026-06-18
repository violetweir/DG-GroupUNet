import math

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_

__all__ = ["EGEUNet"]


class LayerNorm(nn.Module):
    """LayerNorm that supports both channels_last and channels_first tensors."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GroupAggregationBridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=None):
        super().__init__()
        if d_list is None:
            d_list = [1, 2, 5, 7]

        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.groups = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(normalized_shape=group_size + 1, data_format="channels_first"),
                    nn.Conv2d(
                        group_size + 1,
                        group_size + 1,
                        kernel_size=3,
                        stride=1,
                        padding=(k_size + (k_size - 1) * (d - 1)) // 2,
                        dilation=d,
                        groups=group_size + 1,
                    ),
                )
                for d in d_list
            ]
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format="channels_first"),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=xl.shape[2:], mode="bilinear", align_corners=True)
        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl, 4, dim=1)
        x = [
            group(torch.cat((xh_part, xl_part, mask), dim=1))
            for group, xh_part, xl_part in zip(self.groups, xh_chunks, xl_chunks)
        ]
        return self.tail_conv(torch.cat(x, dim=1))


class GroupedMultiAxisHadamardProductAttention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.ones(1, c_dim_in, x, y), requires_grad=True)
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1),
        )

        self.params_zx = nn.Parameter(torch.ones(1, 1, c_dim_in, x), requires_grad=True)
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1),
        )

        self.params_zy = nn.Parameter(torch.ones(1, 1, c_dim_in, y), requires_grad=True)
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in),
        )
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format="channels_first")
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_xy = F.interpolate(self.params_xy, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x1 = x1 * self.conv_xy(params_xy)

        x2 = x2.permute(0, 3, 1, 2)
        params_zx = F.interpolate(self.params_zx, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x2 = x2 * self.conv_zx(params_zx.squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)

        x3 = x3.permute(0, 2, 1, 3)
        params_zy = F.interpolate(self.params_zy, size=x3.shape[2:], mode="bilinear", align_corners=True)
        x3 = x3 * self.conv_zy(params_zy.squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.ldw(self.norm2(x))


class EGEUNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        in_channels=None,
        c_list=None,
        bridge=True,
        gt_ds=True,
        return_deep_supervision=False,
    ):
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]
        if in_channels is not None:
            input_channels = in_channels

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.return_deep_supervision = return_deep_supervision

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[4], c_list[5]))

        if bridge:
            self.GAB1 = GroupAggregationBridge(c_list[1], c_list[0])
            self.GAB2 = GroupAggregationBridge(c_list[2], c_list[1])
            self.GAB3 = GroupAggregationBridge(c_list[3], c_list[2])
            self.GAB4 = GroupAggregationBridge(c_list[4], c_list[3])
            self.GAB5 = GroupAggregationBridge(c_list[5], c_list[4])

        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))

        self.decoder1 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(GroupedMultiAxisHadamardProductAttention(c_list[3], c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _make_mask(self, x):
        return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    def _bridge(self, bridge_layer, xh, xl, mask=None):
        if not self.bridge:
            return xl
        if mask is None:
            mask = self._make_mask(xl)
        return bridge_layer(xh, xl, mask)

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        out = F.gelu(self.encoder6(out))
        t6 = out

        deep_outputs = []
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        gt_pre5 = self.gt_conv1(out5) if self.gt_ds else None
        t5 = self._bridge(self.GAB5, t6, t5, gt_pre5)
        out5 = torch.add(out5, t5)
        if gt_pre5 is not None:
            deep_outputs.append(F.interpolate(gt_pre5, scale_factor=32, mode="bilinear", align_corners=True))

        out4 = F.gelu(
            F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        )
        gt_pre4 = self.gt_conv2(out4) if self.gt_ds else None
        t4 = self._bridge(self.GAB4, t5, t4, gt_pre4)
        out4 = torch.add(out4, t4)
        if gt_pre4 is not None:
            deep_outputs.append(F.interpolate(gt_pre4, scale_factor=16, mode="bilinear", align_corners=True))

        out3 = F.gelu(
            F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        )
        gt_pre3 = self.gt_conv3(out3) if self.gt_ds else None
        t3 = self._bridge(self.GAB3, t4, t3, gt_pre3)
        out3 = torch.add(out3, t3)
        if gt_pre3 is not None:
            deep_outputs.append(F.interpolate(gt_pre3, scale_factor=8, mode="bilinear", align_corners=True))

        out2 = F.gelu(
            F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        )
        gt_pre2 = self.gt_conv4(out2) if self.gt_ds else None
        t2 = self._bridge(self.GAB2, t3, t2, gt_pre2)
        out2 = torch.add(out2, t2)
        if gt_pre2 is not None:
            deep_outputs.append(F.interpolate(gt_pre2, scale_factor=4, mode="bilinear", align_corners=True))

        out1 = F.gelu(
            F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        )
        gt_pre1 = self.gt_conv5(out1) if self.gt_ds else None
        t1 = self._bridge(self.GAB1, t2, t1, gt_pre1)
        out1 = torch.add(out1, t1)
        if gt_pre1 is not None:
            deep_outputs.append(F.interpolate(gt_pre1, scale_factor=2, mode="bilinear", align_corners=True))

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        if self.return_deep_supervision:
            return [out0] + deep_outputs
        return [out0]
