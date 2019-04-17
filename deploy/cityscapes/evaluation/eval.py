import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import imageio
import torch
import time
import cv2
import os

from PIL import Image

try:
    from apex.fp16_utils import *
    from apex import amp
    import apex
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def evaluate(data_root, model, result_path, split):
    train_id2label_id = {0: 7,
                         1: 8,
                         2: 11,
                         3: 12,
                         4: 13,
                         5: 17,
                         6: 19,
                         7: 20,
                         8: 21,
                         9: 22,
                         10: 23,
                         11: 24,
                         12: 25,
                         13: 26,
                         14: 27,
                         15: 28,
                         16: 31,
                         17: 32,
                         18: 33}
    mean = [0.2997, 0.3402, 0.3072]
    std = [0.1549, 0.1579, 0.1552]

    trans_norm = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)])

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Inference Model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    data_root = os.path.join(data_root, split)
    org_data_sub = os.listdir(data_root)
    org_data_sub.sort()

    tt_time = time.time()
    for idx in np.arange(len(org_data_sub)):
        city_name = org_data_sub[idx]
        print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
        print("> 2. Processing City # {}...".format(city_name))
        curr_city_path = os.path.join(data_root, city_name)
        images_name = os.listdir(curr_city_path)
        images_name.sort()

        for img_id in np.arange(len(images_name)):
            curr_image = images_name[img_id]
            # print("> # ------------------------------------------------------------------------- #")
            print("> Processing City # {}, Image: {}...".format(city_name, curr_image))

            with torch.no_grad():
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # 2.1 Pre-processing Image
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                curr_img_path = os.path.join(curr_city_path, curr_image)
                image = Image.open(curr_img_path).convert('RGB')
                image = np.array(image, dtype=np.uint8)
                image = np.array(image[:, :, ::-1], dtype=np.uint8)  # From RGB to BGR
                image = trans_norm(image)
                image = torch.unsqueeze(image, dim=0).cuda()  # [N, C, H, W]

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # 2.2 Prediction/Inference
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                start_time = time.time()
                prediction = F.softmax(model(image), dim=1).argmax(dim=1)
                print("> Inference Time: {}s".format(time.time() - start_time))
                prediction = np.squeeze(prediction.cpu().numpy())

                mapper = lambda t: train_id2label_id[t]
                vfunc = np.vectorize(mapper)
                prediction = vfunc(prediction)

                # fun_classes = np.unique(prediction)
                # print('> {} Classes found: {}'.format(len(fun_classes), fun_classes))
                print("> Processed City #{}, Image: {}, Time: {}s".format(city_name, curr_image,
                                                                         (time.time() - start_time)))

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # 2.3 Saving prediction result
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                save_path = os.path.join(result_path, city_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                # cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
                # cv2.imshow("Prediction", prediction)
                # cv2.waitKey(0)

                prediction = prediction.astype(np.uint8)
                save_name = os.path.basename(curr_image)[:-15] + 'pred_labelIds.png'
                save_path = os.path.join(save_path, save_name)
                imageio.imsave(save_path, prediction)

    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Total Time Cost: {}".format(time.time() - tt_time))
    print("> Done!!!")
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


if __name__ == "__main__":
    from models.shufflenetv2plus import ShuffleNetV2Plus
    from models.mobilenetv2plus import MobileNetV2Plus
    from modules.inplace_abn.iabn import InPlaceABNSync
    from functools import partial
    import shutil

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    # Initialize Amp
    amp_handle = amp.init(enabled=True)

    split = "test"
    method = "shufflenetv2plus_x1.0"  # shufflenetv2plus_x1.0  shufflenetv2plus_x0.5  mobilenetv2plus
    data_path = "/home/huijun/Datasets/Cityscapes/leftImg8bit"
    result_path = "/home/huijun/Datasets/Cityscapes/results/{}".format(split)
    weight_path = "/home/huijun/TrainLog/weights/cityscapes_{}_best_model.pkl".format(method)

    if os.path.exists(result_path):
        print("> Path {} is exist, delete it...".format(result_path))
        shutil.rmtree(result_path)

    if not os.path.exists(result_path):
        print("> Path {} is not exist, create it...".format(result_path))
        os.mkdir(result_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    # model = MobileNetV2Plus(num_classes=19, width_multi=1.0, fuse_chns=512,
    #                         aspp_chns=256, aspp_dilate=(12, 24, 36),
    #                         norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))
    model = ShuffleNetV2Plus(num_classes=19, fuse_chns=512, aspp_chns=256,
                             aspp_dilate=(12, 24, 36), width_multi=1.0,
                             norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))
    # model = ShuffleNetV2Plus(num_classes=19, fuse_chns=256,
    #                          aspp_chns=128, aspp_dilate=(12, 24, 36), width_multi=0.5,
    #                          norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))
    model = apex.parallel.convert_syncbn_model(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    pre_weight = torch.load(weight_path)
    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)
    del pre_weight

    model.eval()
    evaluate(data_path, model, result_path, split)

