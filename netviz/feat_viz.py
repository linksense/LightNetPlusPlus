import torch
import torch.nn as nn


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, model, layer_name, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        layers = self.model._modules['module']._modules
        target_layer = layers[self.layer_name]

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)

        self.model(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale:
            self.rescale_output_array(x.size())

        return self.inputs, self.outputs


def data_preprocess(check, mean_check, std_check):
    check = check.astype(np.float32) / 255.0
    check -= mean_check
    check /= std_check

    # HWC -> CHW
    check = check.transpose(2, 0, 1)
    check = np.expand_dims(check, 0)
    check = torch.from_numpy(check).float()

    return check


if __name__ == "__main__":
    import os
    import numpy as np
    from PIL import Image
    import scipy.misc as misc

    from modules.inplace_abn.iabn import InPlaceABNSync
    from models.mobilenetv2plus import MobileNetV2Plus
    from models.shufflenetv2plus import ShuffleNetV2Plus
    import torch.backends.cudnn as cudnn
    from functools import partial
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        from apex.fp16_utils import *
        from apex import amp
        import apex
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    # Initialize Amp
    amp_handle = amp.init(enabled=True)

    root = "/home/huijun/TrainLog/weights"
    dataset = "deepdrive"  # cityscapes  deepdrive
    method = "shufflenetv2plus_x1.0"  # mobilenetv2plus  shufflenetv2plus_x1.0 shufflenetv2plus_x0.5
    # model = MobileNetV2Plus(num_classes=19, width_multi=1.0, fuse_chns=512,
    #                         aspp_chns=256, aspp_dilate=(12, 24, 36),
    #                         norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))

    model = ShuffleNetV2Plus(num_classes=19, fuse_chns=512,
                             aspp_chns=256, aspp_dilate=(12, 24, 36), width_multi=1.0,
                             norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))

    # model = ShuffleNetV2Plus(num_classes=19, fuse_chns=256, aspp_chns=128, aspp_dilate=(12, 24, 36), width_multi=0.5,
    #                          norm_act=partial(InPlaceABNSync, activation="leaky_relu", slope=0.01))
    model = apex.parallel.convert_syncbn_model(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    pre_weight = torch.load(os.path.join(root, "{}_{}_best_model.pkl").format(dataset, method))

    # mean = pre_weight['mean']
    # std = pre_weight['std']
    mean = [0.2997, 0.3402, 0.3072]
    std = [0.1549, 0.1579, 0.1552]
    pre_weight = pre_weight["model_state"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pre_weight.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    layer_name = "usm"
    extractor = HookBasedFeatureExtractor(model=model, layer_name=layer_name).cuda()

    image_root = "/home/huijun/PycharmProjects/LightNet++/netviz/images"
    image_name = "munster_000168_000019_leftImg8bit.png"
    image_path = os.path.join(image_root, image_name)

    net_h, net_w = 1024, 2048
    img_org = Image.open(image_path).convert('RGB')
    img_org = np.array(img_org, dtype=np.uint8).copy()
    img_org_copy = img_org.copy()
    img_org = np.array(img_org[:, :, ::-1], dtype=np.uint8)

    image = data_preprocess(img_org, mean, std).cuda()
    _, out_feat = extractor.forward(image)

    out_feat = out_feat.view(-1, out_feat.size(2), out_feat.size(3)).cpu().numpy()
    out_feat = np.squeeze(out_feat)

    # save_root = "/home/huijun/PycharmProjects/LightNet++/netviz/feat_vizs"
    # save_root = os.path.join(save_root, method, layer_name, image_name)
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)

    for chn, feat in enumerate(out_feat):
        print("> +++++++++++++++++++++++++++++++++++++++++++++ <")
        print("> Processing Channel: {}...".format(chn))
        feat = 255.0 * ((feat - feat.min().min()) / (feat.max().max() - feat.min().min() + 1e-6))
        feat = feat.astype(np.uint8)
        feat = misc.imresize(feat, (net_h, net_w), interp="bilinear")

        # cv2.namedWindow("ImageOut", cv2.WINDOW_NORMAL)
        # cv2.imshow("ImageOut", feat)
        # cv2.waitKey()

        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(layer_name, fontsize=16)

        axs[0].imshow(img_org_copy)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_title("Base Image")

        axs[1].imshow(feat)
        # pos = axs[1].imshow(img_org_copy, alpha=0.025)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].set_title("HeatMap on Image")
        # fig.colorbar(pos, ax=axs[1])
        # plt.savefig(os.path.join(save_root, str(chn) + "_heatmap_2d.png"))

        plt.show()
