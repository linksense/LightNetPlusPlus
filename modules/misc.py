import torch
import torch.nn as nn

from collections import OrderedDict


class CoordInfo(nn.Module):
    def __init__(self, with_r=True):
        super(CoordInfo, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Add Cartesian Coordination Info to Current Tensor
        :param x: shape(N, C, H, W)
        :return:  shape(N, C+2 or C+3, H, W)
        """
        batch_size, _, height, width = x.size()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Meshgrid using Pytorch
        # i_coords([[[0., 0., 0., 0., 0., 0.],
        #            [1., 1., 1., 1., 1., 1.],
        #            [2., 2., 2., 2., 2., 2.],
        #            [3., 3., 3., 3., 3., 3.]]])
        #
        # j_coords([[[0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.]]])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        i_coords = torch.arange(height).repeat(1, width, 1).transpose(1, 2)  # [1, H, W]
        j_coords = torch.arange(width).repeat(1, height, 1)                  # [1, H, W]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Normalization (-1, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        i_coords = i_coords.float() / (height - 1)
        j_coords = j_coords.float() / (width - 1)

        i_coords = i_coords * 2 - 1
        j_coords = j_coords * 2 - 1

        i_coords = i_coords.repeat(batch_size, 1, 1, 1)  # [N, 1, H, W]
        j_coords = j_coords.repeat(batch_size, 1, 1, 1)  # [N, 1, H, W]

        ret = torch.cat([x, i_coords.type_as(x), j_coords.type_as(x)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(i_coords.type_as(x) - 0.5, 2) + torch.pow(j_coords.type_as(x) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Large Separable Convolution Block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class LightHeadBlock(nn.Module):
    def __init__(self, in_chs, mid_chs=64, out_chs=256, kernel_size=15):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)

        # kernel size had better be odd number so as to avoid alignment error
        self.conv_l = nn.Sequential(OrderedDict([("conv_lu", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0))),
                                                 ("conv_ld", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad)))]))

        self.conv_r = nn.Sequential(OrderedDict([("conv_ru", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad))),
                                                 ("conv_rd", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0)))]))

    def forward(self, x):
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


if __name__ == "__main__":
    import os
    import time

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net_h, net_w = 4, 6
    dummy_in = torch.randn(12, 3, net_h, net_w).requires_grad_()

    co_info = CoordInfo()

    while True:
        start_time = time.time()
        dummy_out = co_info(dummy_in)
        end_time = time.time()
        print("Inference time: {}s".format(end_time - start_time))
