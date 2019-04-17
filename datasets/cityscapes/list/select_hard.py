import os
import numpy as np
from PIL import Image


if __name__ == "__main__":
    data_root = "/home/huijun/Datasets/Cityscapes"
    list_path = "/home/huijun/PycharmProjects/LightNet++/datasets/cityscapes/list/train+.lst"
    image_paths = [i_id.strip() for i_id in open("{}".format(list_path))]

    list_save = "hard.lst"
    # hard_id = {"wall": 12, "fence": 13, "truck": 27, "train": 31}
    hard_id = {"wall": 12, "truck": 27, "train": 31}

    with open(list_save, 'w') as f:
        for idx, image_path_sub in enumerate(image_paths):
            image_path = os.path.join(data_root, image_path_sub)
            mask_path = image_path.replace("leftImg8bit", "gtFine")
            mask_path = mask_path.replace(".png", "_labelIds.png")

            if not os.path.isfile(image_path) or not os.path.exists(image_path):
                raise Exception("> Image: {} is not a file or not exist, can not be opened.".format(image_path))

            if not os.path.isfile(mask_path) or not os.path.exists(mask_path):
                raise Exception("> Mask: {} is not a file or not exist, can not be opened.".format(mask_path))

            # --------------------------------------------------------- #
            # 1. Read Image and Mask
            # --------------------------------------------------------- #
            # image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.uint8).copy()

            fun_classes = np.unique(mask)
            print('> {} Classes found: {}'.format(len(fun_classes), fun_classes))

            has_hard = False
            for key, hid in hard_id.items():
                if hid in fun_classes:
                    has_hard = True
                    print("> {} is hard id, put this image into hard list!!!".format(hid))

            if has_hard:
                f.write(image_path_sub + os.linesep)



