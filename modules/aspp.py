import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from modules.inplace_abn.iabn import InPlaceABN


class ASPPBlock(nn.Module):
    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(4, 8, 12)):
        super(ASPPBlock, self).__init__()
        self.up_ratio = up_ratio

        # --------------------------------------- #
        # 1. For image-level feature
        # --------------------------------------- #
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((3, 3))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("bn1_1", nn.BatchNorm2d(num_features=out_chs))]))

        # --------------------------------------- #
        # 2. Convolution: 1x1
        # --------------------------------------- #
        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1)),
                                                  ("bn1_1", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 3. Convolution: 3x3, dilation: aspp_dilate[0]
        # ------------------------------------------------- #
        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_dilate[0], bias=False,
                                                                          groups=1, dilation=aspp_dilate[0])),
                                                    ("bn2_1", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 4. Convolution: 3x3, dilation: aspp_dilate[1]
        # ------------------------------------------------- #
        self.aspp_bra2 = nn.Sequential(OrderedDict([("conv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_dilate[1], bias=False,
                                                                          groups=1, dilation=aspp_dilate[1])),
                                                    ("bn2_2", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 5. Convolution: 3x3, dilation: aspp_dilate[2]
        # ------------------------------------------------- #
        self.aspp_bra3 = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_dilate[2], bias=False,
                                                                          groups=1, dilation=aspp_dilate[2])),
                                                    ("bn2_3", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 6. down channel after concatenate
        # ------------------------------------------------- #
        self.aspp_catdown = nn.Sequential(OrderedDict([("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
                                                       ("act", nn.LeakyReLU(inplace=True, negative_slope=0.1)),
                                                       ("dropout", nn.Dropout2d(p=0.25, inplace=True))]))

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()

        # ------------------------------------------------- #
        # 1. Atrous Spacial Pyramid Pooling
        # ------------------------------------------------- #
        x = torch.cat((self.aspp_bra1(x),
                       F.interpolate(input=self.gave_pool(x), size=(feat_h, feat_w),
                                     mode="bilinear", align_corners=True),
                       self.aspp_bra2(x),
                       self.conv1x1(x),
                       self.aspp_bra3(x)), dim=1)

        # ------------------------------------------------- #
        # 2. up-sampling the feature-map
        # ------------------------------------------------- #
        return F.interpolate(input=self.aspp_catdown(x),
                             size=(int(feat_h * self.up_ratio),
                                   int(feat_w * self.up_ratio)),
                             mode="bilinear", align_corners=True)


class ASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(12, 24, 36), norm_act=InPlaceABN):
        super(ASPPInPlaceABNBlock, self).__init__()
        self.up_ratio = up_ratio

        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((3, 3))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1))]))

        self.conv1x1 = nn.Conv2d(in_chs, out_chs,
                                 kernel_size=1, stride=1, padding=0,
                                 groups=1, dilation=1, bias=False)

        self.aspp_bra1 = nn.Conv2d(in_chs, out_chs,
                                   kernel_size=3, stride=1, padding=aspp_dilate[0],
                                   groups=1, dilation=aspp_dilate[0], bias=False)

        self.aspp_bra2 = nn.Conv2d(in_chs, out_chs,
                                   kernel_size=3, stride=1, padding=aspp_dilate[1],
                                   groups=1, dilation=aspp_dilate[1], bias=False)

        self.aspp_bra3 = nn.Conv2d(in_chs, out_chs,
                                   kernel_size=3, stride=1, padding=aspp_dilate[2],
                                   groups=1, dilation=aspp_dilate[2], bias=False)

        self.aspp_catdown = nn.Sequential(OrderedDict([("norm_act", norm_act(5*out_chs)),
                                                       ("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("dropout", nn.Dropout2d(p=0.25, inplace=True))]))

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()

        # ------------------------------------------------- #
        # 1. Atrous Spacial Pyramid Pooling
        # ------------------------------------------------- #
        x = self.in_norm(x)
        x = torch.cat((self.aspp_bra1(x),
                       F.interpolate(input=self.gave_pool(x), size=(feat_h, feat_w),
                                     mode="bilinear", align_corners=True),
                       self.aspp_bra2(x),
                       self.conv1x1(x),
                       self.aspp_bra3(x)), dim=1)
        # ------------------------------------------------- #
        # 2. up-sampling the feature-map
        # ------------------------------------------------- #
        return F.interpolate(input=self.aspp_catdown(x),
                             size=(int(feat_h * self.up_ratio),
                                   int(feat_w * self.up_ratio)),
                             mode="bilinear", align_corners=True)


class DSASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(12, 24, 36), norm_act=InPlaceABN):
        super(DSASPPInPlaceABNBlock, self).__init__()
        self.up_ratio = up_ratio

        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((3, 3))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, dilation=1,
                                                                          bias=False))]))

        self.conv1x1 = nn.Conv2d(in_chs, out_chs,
                                 kernel_size=1, stride=1, padding=0,
                                 groups=1, dilation=1, bias=False)

        self.aspp_bra1 = nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_dilate[0],
                                   groups=in_chs, dilation=aspp_dilate[0], bias=False)

        self.aspp_bra2 = nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_dilate[1],
                                   groups=in_chs, dilation=aspp_dilate[1], bias=False)

        self.aspp_bra3 = nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_dilate[2],
                                   groups=in_chs, dilation=aspp_dilate[2], bias=False)

        self.aspp_catdown = nn.Sequential(OrderedDict([("norm_act", norm_act(3 * in_chs + 2 * out_chs)),
                                                       ("conv_down", nn.Conv2d(3 * in_chs + 2 * out_chs, out_chs,
                                                                               kernel_size=1, stride=1, padding=1,
                                                                               groups=1, dilation=1, bias=False))]))

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()

        # ------------------------------------------------- #
        # 1. Atrous Spacial Pyramid Pooling
        # ------------------------------------------------- #
        x = self.in_norm(x)
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


class DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, norm_act=InPlaceABN):
        super(DenseAsppBlock, self).__init__()

        self.add_module('norm_1', norm_act(input_num))
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1))
        self.add_module('norm_2', norm_act(num1))
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


# class RFBlock(nn.Module):
#     def __init__(self, in_chs, out_chs, scale=0.1, feat_res=(56, 112), aspp_dilate=(12, 24, 36),
#                  up_ratio=2, norm_act=InPlaceABN):
#         super(RFBlock, self).__init__()
#         self.feat_res = feat_res
#         self.up_ratio = up_ratio
#
#         self.scale = scale
#
#         self.down_chs = nn.Sequential(OrderedDict([("norm_act", norm_act(in_chs)),
#                                                    ("down_conv1x1", nn.Conv2d(in_chs, out_chs,
#                                                                               kernel_size=1, stride=1,
#                                                                               padding=0, bias=False))]))
#
#         self.gave_pool = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
#                                                     ("gavg", nn.AdaptiveAvgPool2d((1, 1))),
#                                                     ("conv1_0", nn.Conv2d(out_chs, out_chs,
#                                                                           kernel_size=1, stride=1, padding=0,
#                                                                           groups=1, bias=False, dilation=1)),
#                                                     ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))
#
#         self.branch0 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
#                                                   ("conv1x1", nn.Conv2d(out_chs, out_chs,
#                                                                         kernel_size=1, stride=1,
#                                                                         padding=0, bias=False)),
#                                                   ("norm_act", norm_act(out_chs)),
#                                                   ("aconv1", nn.Conv2d(out_chs, out_chs,
#                                                                        kernel_size=3, stride=1,
#                                                                        padding=1, dilation=1,
#                                                                        bias=False))]))
#
#         self.branch1 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
#                                                   ("conv1x3", nn.Conv2d(out_chs, (out_chs // 2) * 3,
#                                                                         kernel_size=(1, 3), stride=1,
#                                                                         padding=(0, 1), bias=False)),
#                                                   ("norm_act", norm_act((out_chs // 2) * 3)),
#                                                   ("conv3x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
#                                                                         kernel_size=(3, 1), stride=1,
#                                                                         padding=(1, 0), bias=False)),
#                                                   ("norm_act", norm_act(out_chs)),
#                                                   ("aconv3", nn.Conv2d(out_chs, out_chs,
#                                                                        kernel_size=3, stride=1,
#                                                                        padding=aspp_dilate[0],
#                                                                        dilation=aspp_dilate[0],
#                                                                        bias=False))]))
#
#         self.branch2 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
#                                                   ("conv1x5", nn.Conv2d(out_chs, (out_chs // 2) * 3,
#                                                                         kernel_size=(1, 5), stride=1,
#                                                                         padding=(0, 2), bias=False)),
#                                                   ("norm_act", norm_act((out_chs // 2) * 3)),
#                                                   ("conv5x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
#                                                                         kernel_size=(5, 1), stride=1,
#                                                                         padding=(2, 0), bias=False)),
#                                                   ("norm_act", norm_act(out_chs)),
#                                                   ("aconv5", nn.Conv2d(out_chs, out_chs,
#                                                                        kernel_size=3, stride=1,
#                                                                        padding=aspp_dilate[1],
#                                                                        dilation=aspp_dilate[1],
#                                                                        bias=False))]))
#
#         self.branch3 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
#                                                   ("conv1x7", nn.Conv2d(out_chs, (out_chs // 2) * 3,
#                                                                         kernel_size=(1, 7), stride=1,
#                                                                         padding=(0, 3), bias=False)),
#                                                   ("norm_act", norm_act((out_chs // 2) * 3)),
#                                                   ("conv7x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
#                                                                         kernel_size=(7, 1), stride=1,
#                                                                         padding=(3, 0), bias=False)),
#                                                   ("norm_act", norm_act(out_chs)),
#                                                   ("aconv7", nn.Conv2d(out_chs, out_chs,
#                                                                        kernel_size=3, stride=1,
#                                                                        padding=aspp_dilate[2],
#                                                                        dilation=aspp_dilate[2],
#                                                                        bias=False))]))
#
#         self.conv_linear = nn.Sequential(OrderedDict([("conv1x1_linear", nn.Conv2d(out_chs * 5, out_chs,
#                                                                                    kernel_size=1, stride=1,
#                                                                                    padding=0, bias=False))]))
#
#     def forward(self, x):
#         down = self.down_chs(x)
#         out = torch.cat([self.gave_pool(down.clone()),
#                          self.branch0(down.clone()),
#                          self.branch1(down.clone()),
#                          self.branch2(down.clone()),
#                          self.branch3(down.clone())], dim=1)
#
#         return F.interpolate(input=self.conv_linear(x),
#                              size=(int(self.feat_res[0]*self.up_ratio),
#                                    int(self.feat_res[1]*self.up_ratio)),
#                              mode="bilinear", align_corners=True)
