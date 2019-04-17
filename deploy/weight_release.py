import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch
import time
import os

try:
    from apex.fp16_utils import *
    from apex import amp
    import apex
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    log_path = "/home/huijun/TrainLog"

    dataset = "cityscapes"
    method = "shufflenetv2plus_x1.0"  # shufflenetv2plus_x1.0 shufflenetv2plus_x0.5  mobilenetv2plus
    checkpoint_path = "{}/weights/{}_{}_best_model.pkl".format(log_path, dataset, method)

    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Loading Original Checkpoint...")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    checkpoint = torch.load(checkpoint_path)
    beat_iou = checkpoint['best_iou']
    checkpoint = checkpoint['model_state']

    mean = np.array([0.2997, 0.3402, 0.3072])
    std = np.array([0.1380, 0.1579, 0.1803])

    state = {"model_state": checkpoint,
             "mean": mean,
             "std": std}

    save_path = "{}/release/{}_{}_{:.3f}.pkl".format(log_path, dataset, method, beat_iou)
    torch.save(state, save_path)
    print("> 2. Checkpoint released!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
