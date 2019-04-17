import os
import shutil
import imageio
import numpy as np

from PIL import Image
from utils.utils import recursive_glob


if __name__ == "__main__":
    img_h, img_w = 1024, 2048
    deepdrive_root = "/home/huijun/Datasets/DeepDrive"
    cityscapes_root = "/home/huijun/Datasets/Cityscapes"

    cvt_img_root = os.path.join(cityscapes_root, "leftImg8bit", "deepdrive")
    cvt_msk_root = os.path.join(cityscapes_root, "gtFine", "deepdrive")

    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    if os.path.exists(cvt_img_root):
        print("> Path {} is exist, delete it...".format(cvt_img_root))
        shutil.rmtree(cvt_img_root)
    if os.path.exists(cvt_msk_root):
        print("> Path {} is exist, delete it...".format(cvt_msk_root))
        shutil.rmtree(cvt_msk_root)

    if not os.path.exists(cvt_img_root):
        print("> Path {} is not exist, create it...".format(cvt_img_root))
        os.mkdir(cvt_img_root)
    if not os.path.exists(cvt_msk_root):
        print("> Path {} is not exist, create it...".format(cvt_msk_root))
        os.mkdir(cvt_msk_root)

    img_list = recursive_glob(rootdir=os.path.join(deepdrive_root, "images"), suffix=".jpg")

    for idx, img_path in enumerate(img_list):
        img_name = os.path.basename(img_path)
        msk_name = img_name.replace(".jpg", "_train_id.png")
        msk_path = os.path.join(deepdrive_root, "labels", msk_name)

        print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
        print("> Processing {}...".format(img_name))
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        image = image.resize((img_w, img_h), Image.BILINEAR)
        mask = mask.resize((img_w, img_h), Image.NEAREST)

        image = np.array(image, dtype=np.uint8).copy()
        mask = np.array(mask, dtype=np.uint8).copy()

        save_img_name = "deepdrive_000000_{}_leftImg8bit.png".format(str(idx).zfill(6))
        save_img_path = os.path.join(cvt_img_root, save_img_name)
        save_msk_name = "deepdrive_000000_{}_gtFine_labelIds.png".format(str(idx).zfill(6))
        save_msk_path = os.path.join(cvt_msk_root, save_msk_name)
        imageio.imsave(save_img_path, image)
        imageio.imsave(save_msk_path, mask)
