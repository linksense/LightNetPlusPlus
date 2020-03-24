# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# PyTorch implementation for MixNetSeg
# class:
#       > Swish
#       > SEBlock
#       > GPConv
#       > MDConv
#       > MixDepthBlock
#       > MixNetSeg(S, M, L)
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   15.02.2020
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
from torch.nn import functional as F
from collections import OrderedDict
from kornia import gaussian_blur2d
from torch import nn
import torch
import math


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # 1. Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # 2. Flatten
    x = x.view(batch_size, -1, height, width)

    return x


def usm(x, kernel_size=(7, 7), amount=1.0, threshold=0):
    res = x.clone()

    blurred = gaussian_blur2d(x, kernel_size=kernel_size, sigma=(1.0, 1.0))
    sharpened = res * (amount + 1.0) - amount * blurred

    if threshold > 0:
        sharpened = torch.where(torch.abs(res - blurred) < threshold, sharpened, res)

    return sharpened


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Swish: Swish Activation Function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim, act_type="relu"):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", Swish(inplace=True) if act_type == "swish" else nn.LeakyReLU(inplace=True, negative_slope=0.01)),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
        return torch.mul(x, x_se)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 groups=1, dilate=1, act_type="relu"):
        super(ConvBlock, self).__init__()
        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate

        self.conv_block = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("act", Swish(inplace=True) if act_type == "swish" else nn.LeakyReLU(inplace=True, negative_slope=0.01))
        ]))

    def forward(self, x):
        return self.conv_block(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# GPConv: Grouped Point-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class GPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_sizes):
        super(GPConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_in_dim = in_planes // self.num_groups
        sub_out_dim = out_planes // self.num_groups

        self.group_point_wise = nn.ModuleList()
        for _ in kernel_sizes:
            self.group_point_wise.append(nn.Conv2d(sub_in_dim, sub_out_dim,
                                                   kernel_size=1, stride=1, padding=0,
                                                   groups=1, dilation=1, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.group_point_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.group_point_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MDConv: Mixed Depth-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MDConv(nn.Module):
    def __init__(self, in_planes, kernel_sizes, stride=1, dilate=1):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_hidden_dim = in_planes // self.num_groups

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate

        self.mixed_depth_wise = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = ((kernel_size - 1) // 2) * dilate
            self.mixed_depth_wise.append(nn.Conv2d(sub_hidden_dim, sub_hidden_dim,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   groups=sub_hidden_dim, dilation=dilate, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depth_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.mixed_depth_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MixDepthBlock: MixDepthBlock for MixNetSeg
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MixDepthBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio, exp_kernel_sizes, kernel_sizes, poi_kernel_sizes, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2, act_type="swish"):
        super(MixDepthBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio

        self.groups = len(kernel_sizes)
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = nn.Sequential(OrderedDict([
                ("conv", GPConv(in_planes, hidden_dim, kernel_sizes=exp_kernel_sizes)),
                ("norm", nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)),
                ("act", Swish(inplace=True) if act_type == "swish" else nn.LeakyReLU(inplace=True, negative_slope=0.01))
            ]))

        # step 2. Depth-wise convolution phase
        self.depth_wise = nn.Sequential(OrderedDict([
            ("conv", MDConv(hidden_dim, kernel_sizes=kernel_sizes, stride=stride, dilate=dilate)),
            ("norm", nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)),
            ("act", Swish(inplace=True) if act_type == "swish" else nn.LeakyReLU(inplace=True, negative_slope=0.01))
        ]))

        # step 3. Squeeze and Excitation
        if self.use_se:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            self.se_block = SEBlock(hidden_dim, reduced_dim, act_type=act_type)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", GPConv(hidden_dim, out_planes, kernel_sizes=poi_kernel_sizes)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))

    def forward(self, x):
        res = x

        # step 1. Expansion phase/Point-wise convolution
        if self.expand_ratio != 1:
            x = self.expansion(x)

        # step 2. Depth-wise convolution phase
        x = self.depth_wise(x)

        # step 3. Squeeze and Excitation
        if self.use_se:
            x = self.se_block(x)

        # step 4. Point-wise convolution phase
        x = self.point_wise(x)

        # step 5. Skip connection and drop connect
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)
            x = x + res

        return x


class DSASPPBlock(nn.Module):
    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(6, 12, 18)):
        super(DSASPPBlock, self).__init__()
        self.up_ratio = up_ratio

        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((3, 3))),
                                                    ("conv1_0",
                                                     ConvBlock(in_chs, out_chs, kernel_size=1, stride=1, dilate=1,
                                                               act_type="relu"))]))

        self.conv1x1 = ConvBlock(in_chs, out_chs, kernel_size=1, stride=1,
                                 dilate=1, act_type="relu")
        self.aspp_bra1 = nn.Sequential(
            OrderedDict([("conv", MDConv(in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=aspp_dilate[0])),
                         ("norm", nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.01)),
                         ("act", nn.LeakyReLU(inplace=True, negative_slope=0.01))]))
        self.aspp_bra2 = nn.Sequential(
            OrderedDict([("conv", MDConv(in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=aspp_dilate[1])),
                         ("norm", nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.01)),
                         ("act", nn.LeakyReLU(inplace=True, negative_slope=0.01))]))
        self.aspp_bra3 = nn.Sequential(
            OrderedDict([("conv", MDConv(in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=aspp_dilate[2])),
                         ("norm", nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.01)),
                         ("act", nn.LeakyReLU(inplace=True, negative_slope=0.01))]))

        self.aspp_catdown = ConvBlock((3 * in_chs + 2 * out_chs), out_chs,
                                      kernel_size=1, stride=1, dilate=1, act_type="relu")

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()
        # ------------------------------------------------- #
        # 1. Atrous Spacial Pyramid Pooling
        # ------------------------------------------------- #
        x = self.aspp_catdown(torch.cat((self.aspp_bra1(x),
                                         F.interpolate(input=self.gave_pool(x),
                                                       size=(feat_h, feat_w),
                                                       mode="bilinear",
                                                       align_corners=True),
                                         self.aspp_bra2(x),
                                         self.conv1x1(x),
                                         self.aspp_bra3(x)), dim=1))
        # ------------------------------------------------- #
        # 2. up-sampling the feature-map
        # ------------------------------------------------- #
        return F.interpolate(input=x,
                             size=(int(feat_h * self.up_ratio),
                                   int(feat_w * self.up_ratio)),
                             mode="bilinear", align_corners=True)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, expand_ratio=1, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p1_td = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p2_td = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p3_td = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p4_td = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")

        self.p2_bu = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p3_bu = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p4_bu = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")
        self.p5_bu = MixDepthBlock(feature_size, feature_size, expand_ratio=expand_ratio,
                                   exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                   stride=1, dilate=1, reduction_ratio=2, dropout_rate=0.0, act_type="relu")

        self.w1 = nn.Parameter(torch.Tensor(2, 4).fill_(0.5))
        self.w2 = nn.Parameter(torch.Tensor(3, 4).fill_(0.5))

    def forward(self, inputs):
        p1_x, p2_x, p3_x, p4_x, p5_x = inputs

        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * p5_td)
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * p4_td)
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * p3_td)
        p1_td = self.p1_td(w1[0, 3] * p1_x + w1[1, 3] * F.interpolate(p2_td, scale_factor=2, mode="bilinear", align_corners=True))

        # Calculate Bottom-Up Pathway
        p1_bu = p1_td
        p2_bu = self.p2_bu(
            w2[0, 0] * p2_x + w2[1, 0] * p2_td + w2[2, 0] * F.interpolate(p1_bu, scale_factor=0.5, mode="bilinear", align_corners=True))
        p3_bu = self.p3_bu(w2[0, 1] * p3_x + w2[1, 1] * p3_td + w2[2, 1] * p2_bu)
        p4_bu = self.p4_bu(w2[0, 2] * p4_x + w2[1, 2] * p4_td + w2[2, 2] * p3_bu)
        p5_bu = self.p5_bu(w2[0, 3] * p5_x + w2[1, 3] * p5_td + w2[2, 3] * p4_bu)

        return p1_bu, p2_bu, p3_bu, p4_bu, p5_bu


class BiFPNDecoder(nn.Module):
    def __init__(self, bone_feat_sizes, feature_size=64, expand_ratio=1, fpn_repeats=3):
        super(BiFPNDecoder, self).__init__()
        self.p1 = ConvBlock(bone_feat_sizes[0], feature_size, kernel_size=1, stride=1, act_type="relu")
        self.p2 = ConvBlock(bone_feat_sizes[1], feature_size, kernel_size=1, stride=1, act_type="relu")
        self.p3 = ConvBlock(bone_feat_sizes[2], feature_size, kernel_size=1, stride=1, act_type="relu")
        self.p4 = ConvBlock(bone_feat_sizes[3], feature_size, kernel_size=1, stride=1, act_type="relu")
        self.p5 = ConvBlock(bone_feat_sizes[4], feature_size, kernel_size=1, stride=1, act_type="relu")

        bifpns_seq = []
        for bifpn_id in range(fpn_repeats):
            bifpns_seq.append(("bi_fpn%d" % (bifpn_id + 1), BiFPNBlock(feature_size=feature_size,
                                                                       expand_ratio=expand_ratio)))
        self.bifpns = nn.Sequential(OrderedDict(bifpns_seq))

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        # Calculate the input column of BiFPNDecoder
        return self.bifpns([self.p1(feat1), self.p2(feat2), self.p3(feat3), self.p4(feat4), self.p5(feat5)])


class MixNetSeg(nn.Module):
    def __init__(self, arch="s", decoder_feat=64, fpn_repeats=3, num_classes=19):
        super(MixNetSeg, self).__init__()
        self.num_classes = num_classes
        params = {
            's': (16, [
                # t, c, n, k, ek, pk, s, d, a, se
                [1, 16, 1, [3], [1], [1], 1, 1, "relu", None],

                [6, 24, 1, [3], [1, 1], [1, 1], 2, 1, "relu", None],
                [3, 24, 1, [3], [1, 1], [1, 1], 1, 1, "relu", None],

                [6, 40, 1, [3, 5, 7], [1], [1], 2, 1, "relu", 2],
                [6, 40, 3, [3, 5], [1, 1], [1, 1], 1, 1, "relu", 2],

                [6, 80, 1, [3, 5, 7], [1], [1, 1], 1, 2, "relu", 4],
                [6, 80, 2, [3, 5], [1], [1, 1], 1, 2, "relu", 4],
                [6, 120, 1, [3, 5, 7], [1, 1], [1, 1], 1, 3, "relu", 2],
                [3, 120, 2, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "relu", 2],

                [6, 200, 1, [3, 5, 7, 9, 11], [1], [1], 1, 4, "relu", 2],
                [6, 200, 2, [3, 5, 7, 9], [1], [1, 1], 1, 4, "relu", 2]
            ], 1.0, 1.0, 0.2),
            'm': (24, [
                # t, c, n, k, ek, pk, s, d, a, se
                [1, 24, 1, [3], [1], [1], 1, 1, "relu", None],

                [6, 32, 1, [3, 5, 7], [1, 1], [1, 1], 2, 1, "relu", None],
                [3, 32, 1, [3], [1, 1], [1, 1], 1, 1, "relu", None],

                [6, 40, 1, [3, 5, 7, 9], [1], [1], 2, 1, "relu", 2],
                [6, 40, 3, [3, 5], [1, 1], [1, 1], 1, 1, "relu", 2],

                [6, 80, 1, [3, 5, 7], [1], [1], 1, 2, "relu", 4],
                [6, 80, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 2, "relu", 4],
                [6, 120, 1, [3], [1], [1], 1, 3, "relu", 2],
                [3, 120, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "relu", 2],

                [6, 200, 1, [3, 5, 7, 9], [1], [1], 1, 4, "relu", 2],
                [6, 200, 3, [3, 5, 7, 9], [1], [1, 1], 1, 4, "relu", 2]
            ], 1.0, 1.0, 0.25),
            'l': (24, [
                # t, c, n, k, ek, pk, s, d, a, se
                [1, 24, 1, [3], [1], [1], 1, 1, "relu", None],

                [6, 32, 1, [3, 5, 7], [1, 1], [1, 1], 2, 1, "relu", None],
                [3, 32, 1, [3], [1, 1], [1, 1], 1, 1, "relu", None],

                [6, 40, 1, [3, 5, 7, 9], [1], [1], 2, 1, "relu", 2],
                [6, 40, 3, [3, 5], [1, 1], [1, 1], 1, 1, "relu", 2],

                [6, 80, 1, [3, 5, 7], [1], [1], 1, 2, "relu", 4],
                [6, 80, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 2, "relu", 4],
                [6, 120, 1, [3], [1], [1], 1, 3, "relu", 2],
                [3, 120, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "relu", 2],

                [6, 200, 1, [3, 5, 7, 9], [1], [1], 1, 4, "relu", 2],
                [6, 200, 3, [3, 5, 7, 9], [1], [1, 1], 1, 4, "relu", 2]
            ], 1.3, 1.0, 0.25),
        }

        stem_planes, settings, width_multi, depth_multi, self.dropout_rate = params[arch]
        out_channels = self._round_filters(stem_planes, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2,
                              groups=1, dilate=1, act_type="relu")

        in_channels = out_channels
        mod_id = 0
        for t, c, n, k, ek, pk, s, d, a, se in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            # Create blocks for module
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d if stride == 1 else 1

                blocks.append(("block%d" % (block_id + 1), MixDepthBlock(in_channels, out_channels,
                                                                         expand_ratio=t, exp_kernel_sizes=ek,
                                                                         kernel_sizes=k, poi_kernel_sizes=pk,
                                                                         stride=stride, dilate=dilate,
                                                                         reduction_ratio=se,
                                                                         dropout_rate=0.0,
                                                                         act_type=a)))

                in_channels = out_channels
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        org_last_planes = (
                settings[0][1] + settings[2][1] + settings[4][1] + settings[6][1] + settings[8][1] + settings[10][
            1])
        last_feat = 256
        self.feat_fuse = MixDepthBlock(org_last_planes, last_feat,
                                       expand_ratio=3, exp_kernel_sizes=[1],
                                       kernel_sizes=[3, 5, 7, 9], poi_kernel_sizes=[1],
                                       stride=1, dilate=1, reduction_ratio=1, dropout_rate=0.0, act_type="relu")

        self.bifpn_decoder = BiFPNDecoder(bone_feat_sizes=[settings[2][1], settings[4][1],
                                                           settings[6][1], settings[8][1], last_feat],
                                          feature_size=decoder_feat, expand_ratio=2, fpn_repeats=fpn_repeats)

        self.aux_head = nn.Conv2d(last_feat, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.cls_head = nn.Conv2d(decoder_feat, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.10, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_divisible(value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        if width_multi == 1.0:
            return filters
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    @staticmethod
    def usm(x, kernel_size=(7, 7), amount=1.0, threshold=0):
        res = x.clone()

        blurred = gaussian_blur2d(x, kernel_size=kernel_size, sigma=(1.0, 1.0))
        sharpened = res * (amount + 1.0) - amount * blurred

        if threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < threshold, sharpened, res)

        return F.relu(sharpened, inplace=True)

    def forward(self, x):
        _, _, in_h, in_w = x.size()
        assert (in_h % 32 == 0 and in_w % 32 == 0), "> in_size must product of 32!!!"

        feat1 = self.mod2(self.mod1(x))  # (N, C,   H/2, W/2)
        feat1_1 = F.max_pool2d(input=feat1, kernel_size=3, stride=2, padding=1)

        feat2 = self.mod4(self.mod3(feat1))  # (N, C,   H/4, W/4)
        feat3 = self.mod6(self.mod5(feat2))  # (N, C,   H/8, W/8) 1
        feat4 = self.mod8(self.mod7(feat3))  # (N, C,   H/8, W/8) 2
        feat5 = self.mod10(self.mod9(feat4))  # (N, C,   H/8, W/8) 3
        feat6 = self.mod12(self.mod11(feat5))  # (N, C,   H/8, W/8) 4

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        feat = self.feat_fuse(torch.cat([feat4, F.max_pool2d(input=feat1_1, kernel_size=3, stride=2, padding=1),
                                         feat3, feat6, F.max_pool2d(input=feat2, kernel_size=3, stride=2, padding=1),
                                         feat5], dim=1))
        feat = feat + F.interpolate(F.adaptive_avg_pool2d(feat, output_size=(3, 3)),
                                    size=(feat.size(2), feat.size(3)), mode="bilinear", align_corners=True)
        aux_score = self.aux_head(feat)

        # compute contrast feature
        feat_de2, feat_de3, feat_de4, feat_de5, feat_de = self.bifpn_decoder(feat2, feat3, feat4, feat5, feat)
        feat_final = feat_de2 + F.interpolate((feat_de3 + feat_de4 + feat_de5 + feat_de),
                                              scale_factor=2, mode="bilinear", align_corners=True)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier: pixel-wise classification-segmentation
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        main_score = self.cls_head(feat_final)
        main_score = F.interpolate(input=main_score, size=(in_h, in_w), mode="bilinear", align_corners=True)
        aux_score = F.interpolate(input=aux_score, size=(in_h, in_w), mode="bilinear", align_corners=True)
        return aux_score, main_score


if __name__ == '__main__':
    import os
    from torchstat import stat
    net_h, net_w = 512, 1024
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = MixNetSeg(arch="s", decoder_feat=64, fpn_repeats=3, num_classes=19)
    stat(model, (3, net_h, net_w))
    # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    #
    # model.eval()
    # with torch.no_grad():
    #     while True:
    #         dummy_in = torch.randn(2, 3, net_h, net_w).cuda()
    #         start_time = time.time()
    #         dummy_out = model(dummy_in)
    #         torch.cuda.synchronize()
    #         del dummy_out
    #
    #         print("> Inference Time: {}".format(time.time() - start_time))
