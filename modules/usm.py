import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.inplace_abn.iabn import ABN


def gauss_kernel(kernel_size, sigma):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid.float() - mean) ** 2, dim=-1) / (2. * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depth-wise convolution weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)  # [1, 1 , H, W]
    return gaussian_kernel


# class UnsharpMask(nn.Module):
#     def __init__(self, channels, padding=3, amount=1.0, threshold=0, norm_act=ABN):
#         super(UnsharpMask, self).__init__()
#         self.channels = channels
#         self.padding = padding
#
#         self.amount = amount
#         self.threshold = threshold
#
#         self.norm_act = norm_act(channels)
#
#     def forward(self, x, gauss_filter):
#         x = self.norm_act(x)
#         res = x.clone()
#
#         gauss_filter = gauss_filter.repeat(self.channels, 1, 1, 1)
#         blurred = F.conv2d(input=x, weight=gauss_filter, stride=1, padding=self.padding, groups=x.size(1), bias=None)
#
#         sharpened = res * (self.amount + 1.0) - blurred * self.amount
#
#         if self.threshold > 0:
#             sharpened = torch.where(torch.abs(res - blurred) < self.threshold, sharpened, res)
#
#         return sharpened  # , res - blurred


class UnsharpMaskV2(nn.Module):
    def __init__(self, channel, kernel_size=7, padding=3, amount=1.0, threshold=0, norm_act=ABN):
        super(UnsharpMaskV2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.amount = amount
        self.threshold = threshold
        self.norm_act = norm_act(channel)

    def forward(self, x):
        x = self.norm_act(x)
        res = x.clone()

        blurred = F.avg_pool2d(input=x, kernel_size=self.kernel_size, stride=1, padding=self.padding,
                               ceil_mode=False, count_include_pad=True)

        sharpened = res * (self.amount + 1.0) - blurred * self.amount

        if self.threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < self.threshold, sharpened, res)

        return sharpened  # , res - blurred


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size=11, padding=5, sigma=1.6):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.sigma = sigma

        weights = self.calculate_weights()
        self.register_buffer('gaussian_filter', weights)

    def calculate_weights(self):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma ** 2

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid.float() - mean) ** 2, dim=-1) / (2. * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        return gaussian_kernel

    def forward(self, x):
        return F.conv2d(input=x, weight=self.gaussian_filter,
                        stride=1, padding=self.padding, groups=x.size(1), bias=None)


class UnsharpMask(nn.Module):
    def __init__(self, channels, kernel_size=11, padding=5, sigma=1.0, amount=1.0, threshold=0, norm_act=ABN):
        super(UnsharpMask, self).__init__()
        self.amount = amount
        self.threshold = threshold
        self.norm_act = norm_act(channels)

        self.gauss_blur = GaussianBlur(channels=channels, kernel_size=kernel_size, padding=padding, sigma=sigma)

    def forward(self, x):
        x = self.norm_act(x)

        res = x.clone()
        blurred = self.gauss_blur(x)

        sharpened = res * (self.amount + 1.0) - blurred * self.amount

        if self.threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < self.threshold, sharpened, res)

        return sharpened


if __name__ == "__main__":
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt

    image = imageio.imread("/home/liuhuijun/PycharmProjects/LightNet++/deploy/cityscapes/examples/images/munster_000168_000019_leftImg8bit.png")
    # image = np.array(image[:, :, ::-1], dtype=np.uint8)
    img_copy = image.copy()

    image = image.transpose(2, 0, 1)  # From HWC to CHW (For PyTorch we use N*C*H*W tensor)
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, dim=0).cuda()  # [N, C, H, W]

    usm = UnsharpMask(channels=3, kernel_size=11, padding=5, sigma=1.6).cuda()
    usm.eval()

    with torch.no_grad():
        dummy_out = usm(image)
        dummy_out = np.squeeze(dummy_out.cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
        # mask = np.squeeze(mask.cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
        # blur = np.squeeze(blur.cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)

        fig, axs = plt.subplots(ncols=3, figsize=(13.5, 6))
        axs[0].imshow(img_copy)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_title("Org Image")

        axs[1].imshow(dummy_out)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].set_title("Sharpened Image")

        # axs[2].imshow(mask, cmap="gray")
        # axs[2].get_xaxis().set_visible(False)
        # axs[2].get_yaxis().set_visible(False)
        # axs[2].set_title("Mask Image")

        plt.tight_layout()
        plt.show()
