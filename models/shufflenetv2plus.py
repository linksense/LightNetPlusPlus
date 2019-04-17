import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from collections import OrderedDict

from modules.attentions import PBCSABlock
from modules.aspp import DSASPPInPlaceABNBlock
from modules.usm import UnsharpMask, UnsharpMaskV2
from modules.shuffle import ShuffleRes, ShuffleResIABN
from modules.inplace_abn.iabn import InPlaceABN, InPlaceABNSync, ABN


class ShuffleNetV2Plus(nn.Module):
    def __init__(self, num_classes=19,
                 fuse_chns=512,
                 aspp_chns=256,
                 aspp_dilate=(12, 24, 36),
                 width_multi=1.0,
                 norm_act=InPlaceABN):
        super(ShuffleNetV2Plus, self).__init__()
        self.stg_repeats = [4, 8, 4]
        self.stride = [2, 1, 1]
        self.dilate = [1, 2, 4]

        if width_multi == 0.5:
            self.stg_chns = [-1, 24, 48, 96, 192]
        elif width_multi == 1.0:
            self.stg_chns = [-1, 24, 116, 232, 464]
        elif width_multi == 1.5:
            self.stg_chns = [-1, 24, 176, 352, 704]
        elif width_multi == 2.0:
            self.stg_chns = [-1, 24, 224, 488, 976]
        else:
            raise ValueError(
                """{} width_multi is not supported""".format(width_multi))

        # building first layer
        input_channel = self.stg_chns[1]
        self.mod1 = nn.Sequential(OrderedDict([("conv", nn.Conv2d(3, input_channel,
                                                                  kernel_size=3, stride=2, padding=1,
                                                                  bias=False)),
                                               ("norm", nn.BatchNorm2d(input_channel)),
                                               ("act", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                                               ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        self.features = []
        mod_id = 0
        # building inverted residual blocks
        for stg_idx in range(len(self.stg_repeats)):
            num_repeat = self.stg_repeats[stg_idx]
            stride = self.stride[stg_idx]
            dilate = self.dilate[stg_idx]

            output_channel = self.stg_chns[stg_idx + 2]

            # Create blocks for module
            blocks = []
            for block_id in range(num_repeat):
                if block_id == 0:
                    blocks.append(("block%d" % (block_id + 1), ShuffleRes(input_channel, output_channel,
                                                                          stride=stride, dilate=1, branch_model=2)))

                else:
                    blocks.append(("block%d" % (block_id + 1), ShuffleRes(input_channel, output_channel,
                                                                          stride=1, dilate=dilate, branch_model=1)))
                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        feat_chns = self.stg_chns[1] + self.stg_chns[2] + self.stg_chns[3] + self.stg_chns[4]
        self.feat_fusion = ShuffleResIABN(feat_chns, fuse_chns, stride=1,
                                          dilate=2, branch_model=2, norm_act=norm_act)

        # self.dsn = nn.Sequential(OrderedDict([("norm", norm_act(fuse_chns)),
        #                                       ("conv", nn.Conv2d(fuse_chns, num_classes,
        #                                                          kernel_size=1, stride=1,
        #                                                          padding=0, bias=True))]))

        self.pyramid_pool = DSASPPInPlaceABNBlock(fuse_chns, aspp_chns, aspp_dilate=aspp_dilate, norm_act=norm_act)

        feat_chns = aspp_chns + self.stg_chns[1]
        self.final_fusion = ShuffleResIABN(feat_chns, aspp_chns, stride=1,
                                           dilate=2, branch_model=2, norm_act=norm_act)
        # self.usm = UnsharpMask(aspp_chns, kernel_size=9, padding=4,
        #                        sigma=1.0, amount=1.0, threshold=0, norm_act=norm_act)
        self.usm = UnsharpMaskV2(aspp_chns, kernel_size=9, padding=4, amount=1.0, threshold=0, norm_act=norm_act)

        self.aspp_scse = PBCSABlock(in_chns=aspp_chns, reduct_ratio=16, dilation=4, use_res=True)

        self.score = nn.Sequential(OrderedDict([("dropout", nn.Dropout2d(0.175)),
                                                ("norm", norm_act(aspp_chns)),
                                                ("conv", nn.Conv2d(aspp_chns, num_classes,
                                                                   kernel_size=1, stride=1,
                                                                   padding=0, bias=True))]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN) or \
                    isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                init.normal_(m.weight, 1.0, 0.0256)
                init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, .1)
                init.constant_(m.bias, 0.)

    def train(self, mode=True, freeze_bn=False, freeze_bn_affine=False):
        super(ShuffleNetV2Plus, self).train()
        # if freeze_bn:
        #     print("> Freezing Mean/Var of BatchNorm2D.")
        #     if freeze_bn_affine:
        #         print("> Freezing Weight/Bias of BatchNorm2D.")
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def forward(self, x):
        _, _, in_h, in_w = x.size()
        assert (in_h % 8 == 0 and in_w % 8 == 0), "> Error, in_size must be product of 8!!!"

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(x)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)  # 1/8
        stg2 = self.mod2(stg1)
        stg3 = self.mod3(stg2)
        stg4 = self.mod4(stg3)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        feat = self.feat_fusion(torch.cat((stg2, stg4, stg3, stg1_1), dim=1))
        feat = self.aspp_scse(self.usm(self.final_fusion(torch.cat((stg1, self.pyramid_pool(feat)), dim=1))))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier: pixel-wise classification-segmentation
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        feat = F.interpolate(input=self.score(feat), size=(in_h, in_w), mode="bilinear", align_corners=True)
        return feat


if __name__ == "__main__":
    import os
    import time

    model = ShuffleNetV2Plus(num_classes=19, fuse_chns=512, aspp_chns=256, width_multi=1.0, norm_act=InPlaceABN).cuda()
    model.eval()

    with torch.no_grad():
        while True:
            dummy_in = torch.randn(1, 3, 768, 768).cuda()
            start_time = time.time()
            dummy_out = model(dummy_in)
            dummy_out = F.softmax(dummy_out, dim=1).argmax(dim=1)
            print("> Inference Time: {}".format(time.time() - start_time))
            del dummy_out

