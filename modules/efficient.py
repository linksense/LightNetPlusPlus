# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# PyTorch implementation for EfficientNet
# class:
#       > Swish
#       > SEBlock
#       > MBConvBlock
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   08.02.2020
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
from collections import OrderedDict

import torch.nn.functional as F
import torch.nn as nn
import torch


class DSConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilate=1):
        super(DSConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate
        self.depth_wise = nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_planes, in_planes,
                                                                        kernel_size, stride, padding, dilate,
                                                                        groups=in_planes, bias=False)),
                                                     ("norm", nn.BatchNorm2d(num_features=out_planes, eps=1e-3, momentum=0.01))
                                                     ]))
        self.point_wise = nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_planes, out_planes,
                                                                        kernel_size=1, stride=1, padding=0,
                                                                        dilation=1, groups=1, bias=False)),
                                                     ("norm", nn.BatchNorm2d(num_features=out_planes, eps=1e-3, momentum=0.01)),
                                                     ("act", nn.LeakyReLU(negative_slope=0.01, inplace=True))
                                                     ]))

    def forward(self, x):
        return self.point_wise(self.depth_wise(x))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Swish: Swish Activation Function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, dilate=1):

        super(ConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate

        self.conv_block = nn.Sequential(OrderedDict([
           ("conv", nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("act", nn.LeakyReLU(negative_slope=0.01, inplace=True))
        ]))

    def forward(self, x):
        return self.conv_block(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
        return torch.mul(x, x_se)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p1_td = DSConvBlock(feature_size, feature_size)
        self.p2_td = DSConvBlock(feature_size, feature_size)
        self.p3_td = DSConvBlock(feature_size, feature_size)
        self.p4_td = DSConvBlock(feature_size, feature_size)

        self.p2_out = DSConvBlock(feature_size, feature_size)
        self.p3_out = DSConvBlock(feature_size, feature_size)
        self.p4_out = DSConvBlock(feature_size, feature_size)
        self.p5_out = DSConvBlock(feature_size, feature_size)

        self.w1 = nn.Parameter(torch.Tensor(2, 4).fill_(0.5))
        self.w2 = nn.Parameter(torch.Tensor(3, 4).fill_(0.5))

    def forward(self, inputs):
        p1_x, p2_x, p3_x, p4_x, p5_x = inputs

        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * F.interpolate(p5_td, scale_factor=2, mode="bilinear", align_corners=True))
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * F.interpolate(p4_td, scale_factor=2, mode="bilinear", align_corners=True))
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * F.interpolate(p3_td, scale_factor=2, mode="bilinear", align_corners=True))
        p1_td = self.p1_td(w1[0, 3] * p1_x + w1[1, 3] * F.interpolate(p2_td, scale_factor=2, mode="bilinear", align_corners=True))

        # Calculate Bottom-Up Pathway
        p1_out = p1_td
        p2_out = self.p2_out(w2[0, 0] * p2_x + w2[1, 0] * p2_td + w2[2, 0] * F.interpolate(p1_out, scale_factor=0.5, mode="bilinear", align_corners=True))
        p3_out = self.p3_out(w2[0, 1] * p3_x + w2[1, 1] * p3_td + w2[2, 1] * F.interpolate(p2_out, scale_factor=0.5, mode="bilinear", align_corners=True))
        p4_out = self.p4_out(w2[0, 2] * p4_x + w2[1, 2] * p4_td + w2[2, 2] * F.interpolate(p3_out, scale_factor=0.5, mode="bilinear", align_corners=True))
        p5_out = self.p5_out(w2[0, 3] * p5_x + w2[1, 3] * p5_td + w2[2, 3] * F.interpolate(p4_out, scale_factor=0.5, mode="bilinear", align_corners=True))

        return p1_out, p2_out, p3_out, p4_out, p5_out


class BiFPNDecoder(nn.Module):
    def __init__(self, bone_feat_sizes, feature_size=64, fpn_repeats=2):
        super(BiFPNDecoder, self).__init__()
        self.p1 = ConvBlock(bone_feat_sizes[0], feature_size, kernel_size=1, stride=1)
        self.p2 = ConvBlock(bone_feat_sizes[1], feature_size, kernel_size=1, stride=1)
        self.p3 = ConvBlock(bone_feat_sizes[2], feature_size, kernel_size=1, stride=1)
        self.p4 = ConvBlock(bone_feat_sizes[3], feature_size, kernel_size=1, stride=1)
        self.p5 = ConvBlock(bone_feat_sizes[4], feature_size, kernel_size=1, stride=1)

        bifpns_seq = []
        for bifpn_id in range(fpn_repeats):
            bifpns_seq.append(("bi_fpn%d" % (bifpn_id + 1), BiFPNBlock(feature_size)))
        self.bifpns = nn.Sequential(OrderedDict(bifpns_seq))

    def forward(self, feat1, feat2, feat3, feat4, feat5):

        # Calculate the input column of BiFPNDecoder
        p1 = self.p1(feat1)
        p2 = self.p2(feat2)
        p3 = self.p3(feat3)
        p4 = self.p4(feat4)
        p5 = self.p5(feat5)

        return self.bifpns([p1, p2, p3, p4, p5])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MBConvBlock: MBConvBlock for EfficientSeg
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio,  kernel_size, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = ConvBlock(in_planes, hidden_dim, 1)

        # step 2. Depth-wise convolution phase
        self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
                                    stride=stride, groups=hidden_dim, dilate=dilate)
        # step 3. Squeeze and Excitation
        if self.use_se:
            self.se_block = SEBlock(hidden_dim, reduced_dim)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(hidden_dim, out_planes, kernel_size=1,
                               stride=1, padding=0, dilation=1, groups=1, bias=False)),
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
