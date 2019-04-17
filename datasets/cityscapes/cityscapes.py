# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# DataReader for Cityscapes Dataset
#
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   10.10.2018
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
import torchvision.transforms as transforms
from random import shuffle
import numpy as np
import torch
import os

from PIL import Image, ImageEnhance, ImageFilter
from datasets.augmentations import *
from torch.utils import data


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Lighting data augmentation take from:
# https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Cityscapes(data.Dataset):
    """
        Data Reader for Cityscapes Dataset

        https://www.cityscapes-dataset.com

        Data is derived from CityScapes, and can be downloaded from here:
        https://www.cityscapes-dataset.com/downloads/

        Many Thanks to @fvisin for the loader repo:
        https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
        """
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, data_root, list_path, num_classes=19, phase="train",
                 augmentations=None, use_transform=True, use_lighting=True,
                 mean=[0.2997, 0.3402, 0.3072], std=[0.1549, 0.1579, 0.1552]):
        super(Cityscapes, self).__init__()
        self.data_root = data_root
        self.num_classes = num_classes
        self.phase = phase

        self.augmentations = augmentations
        self.use_transform = use_transform
        self.use_lighting = use_lighting

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.aug_color = transforms.ColorJitter(brightness=0.125, contrast=0.375, saturation=0.275, hue=0.125)

        imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        self.trans_lighting = transforms.Compose([transforms.ToTensor(),
                                                  Lighting(0.1, imagenet_pca['eigval'],
                                                           imagenet_pca['eigvec']),
                                                  transforms.Normalize(mean=mean, std=std)])

        self.trans_norm = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])

        self.image_paths = [i_id.strip() for i_id in open("{}".format(list_path))]
        shuffle(self.image_paths)

        if not self.image_paths:
            raise Exception("> No files found in %s" % os.path.basename(list_path))

        print("> Found %d %s images..." % (len(self.image_paths), os.path.basename(list_path)))

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __len__(self):
        """__len__"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # line = self.files[self.split][index]
        # splits = line.split(" ")
        # image_path, mask_path = os.path.join(self.data_root, splits[0]), os.path.join(self.data_root, splits[1])
        image_path = os.path.join(self.data_root, self.image_paths[index])
        mask_path = image_path.replace("leftImg8bit", "gtFine")
        mask_path = mask_path.replace(".png", "_labelIds.png")

        if not os.path.isfile(image_path) or not os.path.exists(image_path):
            raise Exception("> Image: {} is not a file or not exist, can not be opened.".format(image_path))

        if not os.path.isfile(mask_path) or not os.path.exists(mask_path):
            raise Exception("> Mask: {} is not a file or not exist, can not be opened.".format(mask_path))

        # --------------------------------------------------------- #
        # 1. Read Image and Mask
        # --------------------------------------------------------- #
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if not ("deepdrive" in os.path.basename(mask_path)):
            mask = Image.fromarray(self.encode_segmap(np.array(mask, dtype=np.uint8)), mode='L')

        # --------------------------------------------------------- #
        # 2. Data Augmentation used in training phase
        # --------------------------------------------------------- #
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        # --------------------------------------------------------- #
        # 3. Image Transformation
        # --------------------------------------------------------- #
        # 3.1 Image Color Jitter
        if (self.aug_color is not None) and random.random() < 0.5 and self.phase == "train":
            b_scale = random.uniform(0.125, 0.45)
            c_scale = random.uniform(0.125, 0.45)
            s_scale = random.uniform(0.125, 0.45)
            h_scale = random.uniform(-0.325, 0.325)
            self.aug_color = transforms.ColorJitter(brightness=b_scale,
                                                    contrast=c_scale,
                                                    saturation=s_scale,
                                                    hue=h_scale)

            # 3.2 Sharpen/GaussianBlur
        if random.random() < 0.5 and self.phase == "train":
            scale = random.uniform(1.125, 2.0)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(scale)

        if random.random() < 0.5 and self.phase == "train":
            scale = random.uniform(1.25, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=scale))

        image = np.array(image, dtype=np.uint8).copy()
        mask = np.array(mask, dtype=np.uint8).copy()

        # 3.2 Image transformation
        image = np.array(image[:, :, ::-1], dtype=np.uint8)  # From RGB to BGR
        if self.use_transform:
            if self.use_lighting and random.random() < 0.5 and self.phase == "train":
                image = self.trans_lighting(image)
            else:
                image = self.trans_norm(image)

            return image, torch.from_numpy(mask).long()
        else:
            # --------------------------------- #
            # Only for DataLoader test
            # --------------------------------- #
            image = image.transpose(2, 0, 1)  # From HWC to CHW (For PyTorch we use N*C*H*W tensor)
            image = image.astype(np.uint8)
            return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


# +++++++++++++++++++++++++++++++++++++++++++++ #
# Test the code of 'CityscapesReader'
# +++++++++++++++++++++++++++++++++++++++++++++ #
if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt

    net_h, net_w = 768, 768
    augment = Compose([RandomHorizontallyFlip(), RandomScale((0.75, 1.25)),
                       RandomRotate(5), RandomCrop((net_h, net_w))])

    local_path = "/home/huijun/Datasets/Cityscapes"
    list_path = "/home/huijun/PycharmProjects/LightNet++/datasets/cityscapes/list/deepdrive.lst"
    reader = Cityscapes(data_root=local_path, list_path=list_path, num_classes=19, phase="train",
                        augmentations=augment, use_transform=False, use_lighting=False,
                        mean=[0.2997, 0.3402, 0.3072],
                        std=[0.1549, 0.1579, 0.1552])

    train_loader = data.DataLoader(dataset=reader, batch_size=1, num_workers=1, shuffle=True)
    for idx, data in enumerate(train_loader):
        print("batch :", idx)
        imgs, msks = data

        imgs = imgs.numpy()  # From PyTorch Tensor to Numpy NArray, From BGR to RGB
        imgs = np.squeeze(imgs.astype(np.uint8))
        imgs = imgs.transpose(1, 2, 0)
        imgs = imgs[:, :, ::-1]

        msks = msks.numpy()
        msks = np.squeeze(msks.astype(np.uint8))

        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))

        axs[0].imshow(imgs)
        # axs[0].imshow(mask, origin='left', alpha=0.225)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_title("Image")

        axs[1].imshow(msks)
        # axs[1].imshow(msks, alpha=0.225)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].set_title("Mask Image")
        plt.tight_layout()

        # plt.savefig(os.path.join(save_root, str(chn) + "_heatmap_2d.png"))
        plt.show()
