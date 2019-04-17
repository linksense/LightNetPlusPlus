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

from modules.inplace_abn.iabn import InPlaceABNSync
from functools import partial
from PIL import Image

try:
    from apex.fp16_utils import *
    from apex import amp
    import apex
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def decode_segmap(pred, label_colours, num_classes):
    r = pred.copy()
    g = pred.copy()
    b = pred.copy()
    for l in range(0, num_classes):
        r[pred == l] = label_colours[l][0]
        g[pred == l] = label_colours[l][1]
        b[pred == l] = label_colours[l][2]

    rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def video_demo(data_root, model, method, result_path, split):
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
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                   'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                   'motorcycle', 'bicycle']

    net_h, net_w, color_bar_w = 1024, 2048, 120
    frame_size = (net_w + color_bar_w, net_h)
    codec = cv2.VideoWriter_fourcc(*'MJPG')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 0. Setup Color Bar
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    color_map = label_colours
    num_classes = 19

    grid_height = int(net_h // num_classes)
    start_pixel = int((net_h % num_classes) / 2)

    color_bar = np.ones((net_h, color_bar_w, 3), dtype=np.uint8) * 128
    for train_id in np.arange(num_classes):
        end_pixel = start_pixel + grid_height
        color_bar[start_pixel:end_pixel, :, :] = color_map[train_id]

        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(color_bar, class_names[train_id + 1],
                    (2, start_pixel + 5 + int(grid_height // 2)),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        start_pixel = end_pixel
    color_bar = color_bar[:, :, ::-1]
    my_writer = cv2.VideoWriter("{}_video_demo.avi".format(method), codec, 24.0, frame_size)

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

    for idx in np.arange(len(org_data_sub)):
        city_name = org_data_sub[idx]
        print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
        print("> 2. Processing City # {}...".format(city_name))
        curr_city_path = os.path.join(data_root, city_name)
        images_name = os.listdir(curr_city_path)
        images_name.sort()

        for img_id in np.arange(len(images_name)):
            curr_image = images_name[img_id]
            print("> Processing City #{} Image: {}...".format(city_name, curr_image))

            with torch.no_grad():
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # 2.1 Pre-processing Image
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                curr_img_path = os.path.join(curr_city_path, curr_image)
                image = Image.open(curr_img_path).convert('RGB')
                image = np.array(image, dtype=np.uint8)
                image_org = image.copy()
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
                prediction = decode_segmap(prediction, label_colours, num_classes)
                prediction = prediction.astype(np.uint8)

                print("> Processed City #{} Image: {}, Time: {}s".format(city_name, curr_image,
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

                img_msk = cv2.addWeighted(image_org, 0.55, prediction, 0.45, 0)
                img_msk = img_msk[:, :, ::-1]  # RGB
                img_msk_color = np.concatenate((img_msk, color_bar), axis=1)

                cv2.imshow("show", img_msk_color)
                cv2.waitKey(0)
                my_writer.write(img_msk_color)

    my_writer.release()
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Done!!!")
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


if __name__ == "__main__":
    from models.mobilenetv2plus import MobileNetV2Plus
    from models.shufflenetv2plus import ShuffleNetV2Plus

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    # Initialize Amp
    amp_handle = amp.init(enabled=True)

    split = "demoVideo"
    method = "shufflenetv2plus_x1.0"
    data_path = "/home/huijun/Datasets/Cityscapes/leftImg8bit"
    result_path = "/home/huijun/Datasets/Cityscapes/results/{}".format(split)
    weight_path = "/home/huijun/TrainLog/weights/cityscapes_{}_best_model.pkl".format(method)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    # model = MobileNetV2Plus(num_classes=19, width_multi=1.0,
    #                         aspp_chns=256, aspp_dilate=(12, 24, 36),
    #                         norm_act=partial(InPlaceABNSync, eps=1e-05, momentum=0.1,
    #                                          activation="leaky_relu", slope=0.01))

    model = ShuffleNetV2Plus(num_classes=19, fuse_chns=512, aspp_chns=256, aspp_dilate=(12, 24, 36), width_multi=1.0,
                             norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))

    model = apex.parallel.convert_syncbn_model(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    pre_weight = torch.load(weight_path)
    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)
    del pre_weight

    model.eval()
    video_demo(data_path, model, method, result_path, split)

