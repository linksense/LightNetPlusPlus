import torch
import torch.nn as nn
from modules.inplace_abn.iabn import InPlaceABN


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # 1. Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # 2. Flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleRes(nn.Module):
    def __init__(self, in_chns, out_chns, stride, dilate, branch_model):
        super(ShuffleRes, self).__init__()
        self.branch_model = branch_model

        assert stride in [1, 2]
        self.stride = stride

        mid_chns = out_chns // 2

        if self.branch_model == 1:
            self.branch2 = nn.Sequential(
                # step 1. point-wise convolution
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_chns),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),

                # step 2. depth-wise convolution
                nn.Conv2d(mid_chns, mid_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate, groups=mid_chns, bias=False),
                nn.BatchNorm2d(mid_chns),

                # step 3. point-wise convolution
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_chns),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
            )
        else:
            self.branch1 = nn.Sequential(
                # step 1. depth-wise convolution
                nn.Conv2d(in_chns, in_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate,  groups=in_chns, bias=False),
                nn.BatchNorm2d(in_chns),

                # step 2. point-wise convolution
                nn.Conv2d(in_chns, mid_chns, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_chns),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
            )

            self.branch2 = nn.Sequential(
                # step 1. point-wise convolution
                nn.Conv2d(in_chns, mid_chns, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_chns),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),

                # step 2. depth-wise convolution
                nn.Conv2d(mid_chns, mid_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate, groups=mid_chns, bias=False),
                nn.BatchNorm2d(mid_chns),

                # step 3. point-wise convolution
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_chns),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
            )

    def forward(self, x):
        if 1 == self.branch_model:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        elif 2 == self.branch_model:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        return channel_shuffle(out, 2)


class ShuffleResIABN(nn.Module):
    def __init__(self, in_chns, out_chns, stride, dilate, branch_model, norm_act=InPlaceABN):
        super(ShuffleResIABN, self).__init__()
        self.branch_model = branch_model
        self.stride = stride
        assert stride in [1, 2]

        mid_chns = out_chns // 2

        if self.branch_model == 1:
            self.in_norm = norm_act(mid_chns)
            self.branch2 = nn.Sequential(
                # step 1. point-wise convolution
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),

                # step 2. depth-wise convolution
                norm_act(mid_chns),
                nn.Conv2d(mid_chns, mid_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate, groups=mid_chns, bias=False),

                # step 3. point-wise convolution
                norm_act(mid_chns),
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),
            )
        else:
            self.in_norm = norm_act(in_chns)
            self.branch1 = nn.Sequential(
                # step 1. depth-wise convolution
                nn.Conv2d(in_chns, in_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate,  groups=in_chns, bias=False),

                # step 2. point-wise convolution
                norm_act(in_chns),
                nn.Conv2d(in_chns, mid_chns, 1, 1, 0, bias=False),
            )

            self.branch2 = nn.Sequential(
                # step 1. point-wise convolution
                nn.Conv2d(in_chns, mid_chns, 1, 1, 0, bias=False),

                # step 2. depth-wise convolution
                norm_act(mid_chns),
                nn.Conv2d(mid_chns, mid_chns, kernel_size=3, stride=stride, padding=dilate,
                          dilation=dilate, groups=mid_chns, bias=False),

                # step 3. point-wise convolution
                norm_act(mid_chns),
                nn.Conv2d(mid_chns, mid_chns, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        if 1 == self.branch_model:
            x = self.in_norm(x)
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        elif 2 == self.branch_model:
            x = self.in_norm(x)
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        return channel_shuffle(out, groups=2)
