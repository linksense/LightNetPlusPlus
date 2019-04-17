import torch
import torch.nn as nn
import torch.nn.functional as F

from .inplace_abn.iabn import InPlaceABN
from collections import OrderedDict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Large Separable Convolution Block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class LightHeadBlock(nn.Module):
    def __init__(self, in_chs, mid_chs=64, out_chs=256, kernel_size=15, norm_act=InPlaceABN):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)

        # kernel size had better be odd number so as to avoid alignment error
        self.abn = norm_act(in_chs)
        self.conv_l = nn.Sequential(OrderedDict([("conv_lu", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       stride=(1, 1),
                                                                       padding=(pad, 0),
                                                                       bias=False)),
                                                 ("norm", norm_act(mid_chs)),
                                                 ("conv_ld", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       stride=(1, 1),
                                                                       padding=(0, pad),
                                                                       bias=False))]))

        self.conv_r = nn.Sequential(OrderedDict([("conv_ru", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       stride=(1, 1),
                                                                       padding=(0, pad),
                                                                       bias=False)),
                                                 ("norm", norm_act(mid_chs)),
                                                 ("conv_rd", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0),
                                                                       bias=False))]))

    def forward(self, x):
        x = self.abn(x)
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0))]))

    def forward(self, x):
        inputs = x
        chn_se = self.channel_se(x).sigmoid().exp()
        return torch.mul(inputs, chn_se)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SCSEBlock: Spatial-Channel Squeeze & Excitation (SCSE)
#            namely, Spatial-wise and Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16, is_res=True):
        super(SCSEBlock, self).__init__()
        self.is_res = is_res

        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0))]))

        self.spatial_se = nn.Sequential(OrderedDict([("conv", nn.Conv2d(channel, 1, kernel_size=1, stride=1,
                                                                        padding=0, bias=False))]))

    def forward(self, x):
        inputs = x

        chn_se = self.channel_se(x).sigmoid().exp()
        spa_se = self.spatial_se(x).sigmoid().exp()

        if self.is_res:
            torch.mul(torch.mul(inputs, chn_se), spa_se) + inputs

        return torch.mul(torch.mul(inputs, chn_se), spa_se)


class ModifiedSCSEBlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, is_res=True):
        super(ModifiedSCSEBlock, self).__init__()
        self.is_res = is_res

        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_se = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                  kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                  kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(in_chns))

        self.spatial_se = nn.Sequential(nn.Conv2d(in_chns, 1, kernel_size=1, stride=1,
                                                  padding=0, bias=False),
                                        nn.BatchNorm2d(1))

    def forward(self, x):
        res = x

        ch_att = self.channel_se((self.ch_avg_pool(x) + self.ch_max_pool(x)))
        ch_att = torch.mul(x, ch_att.sigmoid().exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        sp_att = torch.mul(x, self.spatial_se(x).sigmoid().exp())

        if self.is_res:
            ch_att + res + sp_att

        return ch_att + sp_att


class SCSABlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, is_res=True):
        super(SCSABlock, self).__init__()
        self.is_res = is_res

        if is_res:
            self.gamma = nn.Parameter(torch.ones(1))
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(in_chns))

        self.sp_conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
                                     nn.BatchNorm2d(1))

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))

        ch_att = torch.mul(x, (avg_p + max_p).sigmoid().exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        ch_avg = torch.mean(ch_att, dim=1, keepdim=True)
        ch_max = torch.max(ch_att, dim=1, keepdim=True)[0]

        sp_att = torch.mul(ch_att, self.sp_conv(torch.cat([ch_avg, ch_max], dim=1)).sigmoid().exp())

        if self.is_res:
            return sp_att + self.gamma * res
        return sp_att


class PBCSABlock(nn.Module):
    """
    Parallel Bottleneck Channel-Spatial Attention Block
    """
    def __init__(self, in_chns, reduct_ratio=16, dilation=4, use_res=True):
        super(PBCSABlock, self).__init__()
        self.use_res = use_res
        #
        # if is_res:
        #     self.gamma = nn.Parameter(torch.ones(1))
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ch_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0))

        self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),

                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, 1, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1))

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        # res = x
        ch_att = self.se_block(self.ch_avg_pool(x) + self.ch_max_pool(x))
        ch_att = torch.mul(x, ch_att.sigmoid().exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        sp_att = torch.mul(x, self.sp_conv(x).sigmoid().exp())

        if self.use_res:
            return sp_att + x + ch_att

        return sp_att + ch_att
    

class PABlock(nn.Module):
    """Position Attention Block"""
    def __init__(self, in_chns, reduct_ratio=8):
        super(PABlock, self).__init__()
        self.in_chns = in_chns

        self.query = nn.Conv2d(in_channels=in_chns, out_channels=in_chns//reduct_ratio,
                               kernel_size=1, stride=1, padding=0)

        self.key = nn.Conv2d(in_channels=in_chns, out_channels=in_chns//reduct_ratio,
                             kernel_size=1, stride=1, padding=0)

        self.value = nn.Conv2d(in_channels=in_chns, out_channels=in_chns,
                               kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, feat_h, feat_w = x.size()

        proj_query = self.query(x).view(batch_size, -1, feat_h * feat_w).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, feat_h * feat_w)
        proj_value = self.value(x).view(batch_size, -1, feat_h * feat_w)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy).permute(0, 2, 1)

        out = torch.bmm(proj_value, attention).view(batch_size, channels, feat_h, feat_w)

        return self.gamma * out + x


class CABlock(nn.Module):
    """Channel Attention Block"""
    def __init__(self):
        super(CABlock, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, feat_h, feat_w = x.size()

        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        proj_value = x.view(batch_size, channels, -1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, dim=-1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)

        out = torch.bmm(attention, proj_value).view(batch_size, channels, feat_h, feat_w)

        return self.gamma * out + x
