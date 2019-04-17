# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# Data Augmentations for Semantic Segmentation
# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   10.10.2018
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
import math
import random
import numbers

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        """

        :param image: <PIL.Image> Image to be augmented, mode='RGB'
        :param mask: <PIL.Image> Mask to be augmented, mode='L'
        :return: image, mask
        """
        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        for aug in self.augmentations:
            image, mask = aug(image, mask)
        return image, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, image, mask):
        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return image, mask
        if w < tw or h < th:
            return image.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return image.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image, mask):
        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        w, h = image.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return image.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, image, mask):
        if random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, image, mask):
        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        return image.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):  # size: (h, w)
        self.size = size

    def __call__(self, image, mask):
        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        w, h = image.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return image, mask

        oh, ow = self.size
        return image.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomScale(object):
    def __init__(self, limit):
        """

        :param limit: <tuple of float> for example: (0.725, 1.25)
        """
        assert isinstance(limit, tuple), "> limit must be a tuple, for example: (0.725, 1.25)"
        self.limit = limit

    def __call__(self, image, mask):
        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * image.size[0])
        h = int(scale * image.size[1])

        return image.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        assert image.size == mask.size, "> The size of Image and Mask mismatch!!!"

        for attempt in range(12):
            area = image.size[0] * image.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= image.size[0] and h <= image.size[1]:
                x1 = random.randint(0, image.size[0] - w)
                y1 = random.randint(0, image.size[1] - h)

                image = image.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (image.size == (w, h))

                return image.resize((self.size, self.size), Image.BILINEAR), \
                       mask.resize((self.size, self.size), Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(image, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return image.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

