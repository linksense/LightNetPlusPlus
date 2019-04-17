import torch
from tqdm import tqdm


def get_mean_and_std(dataloader):
    '''
    Compute the mean and std value of dataset.
    '''
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('> Computing mean and std of images in the dataset..')
    pbar = tqdm(np.arange(len(dataloader)))
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()  # [N, C, H, W]  BGR
            std[i] += inputs[:, i, :, :].std()    # [N, C, H, W]  BGR
        pbar.update(1)

    pbar.close()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    return mean, std


if __name__ == "__main__":
    import time
    import numpy as np
    from datasets.cityscapes.cityscapes import Cityscapes
    from datasets.augmentations import *

    net_h, net_w = 768, 768
    augment = Compose([RandomHorizontallyFlip(), RandomScale((0.75, 1.25)),
                       RandomRotate(5), RandomCrop((net_h, net_w))])

    local_path = "/home/huijun/Datasets/Cityscapes"
    list_path = "/home/huijun/PycharmProjects/LightNet++/datasets/cityscapes/list/train+.lst"
    reader = Cityscapes(data_root=local_path, list_path=list_path, num_classes=19, phase="train",
                        augmentations=augment, use_transform=False, use_lighting=False,
                        mean=[0.41738699, 0.45732192, 0.46886091],
                        std=[0.25685097, 0.26509955, 0.29067996])

    dataloader = torch.utils.data.DataLoader(dataset=reader, batch_size=1, num_workers=20, shuffle=False)

    count = 3
    mmean = torch.zeros(3)
    mstd = torch.zeros(3)
    time_cost = 0.0
    for idx in np.arange(count):
        print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
        print("> Epoch: {}...".format(idx + 1))
        start_time = time.time()
        mean, std = get_mean_and_std(dataloader)
        mmean = mmean + mean
        mstd = mstd + std

        end_time = time.time() - start_time
        time_cost += end_time
        print("> Time: {}..., ".format(end_time))
        print("> Mean (BGR): {}".format(mean / 255.0))
        print("> STD  (BGR): {}".format(std / 255.0))

    print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
    print("> Done, Time: {}s".format(time_cost))
    print("> Mean (BGR): {}".format((mmean / count) / 255.0))
    print("> STD  (BGR): {}".format((mstd / count) / 255.0))
    print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
