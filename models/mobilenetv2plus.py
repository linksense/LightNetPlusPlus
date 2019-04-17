import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from modules.attentions import PBCSABlock
from modules.aspp import DSASPPInPlaceABNBlock
from modules.inplace_abn.iabn import InPlaceABN, InPlaceABNSync, ABN
from modules.mobile import InvertedResidual, InvertedResidualIABN
from collections import OrderedDict


class MobileNetV2Plus(nn.Module):
    def __init__(self, num_classes=19,
                 width_multi=1.0,
                 fuse_chns=512,
                 aspp_chns=256,
                 aspp_dilate=(12, 24, 36),
                 norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param num_classes:    (int)  Number of classes
        :param width_multi: (float) Network width multiplier
        :param aspp_chns:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_dilate:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2Plus, self).__init__()
        self.num_classes = num_classes

        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 1, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 1, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        input_channel = int(32 * width_multi)
        self.mod1 = nn.Sequential(OrderedDict([("conv", nn.Conv2d(3, input_channel,
                                                                  kernel_size=3, stride=2, padding=1,
                                                                  bias=False)),
                                               ("norm", nn.BatchNorm2d(input_channel)),
                                               ("act", nn.LeakyReLU(inplace=True, negative_slope=0.01))]))

        # building inverted residual blocks
        mod_id = 0
        for t, c, n, s, d in self.inverted_residual_setting:
            output_channel = int(c * width_multi)

            # Create blocks for module
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=s,
                                                                                dilate=1,
                                                                                expand_ratio=t)))
                else:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=1,
                                                                                dilate=d,
                                                                                expand_ratio=t)))

                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        # building last several layers
        org_last_chns = (self.inverted_residual_setting[0][1] +
                         self.inverted_residual_setting[1][1] +
                         self.inverted_residual_setting[2][1] +
                         self.inverted_residual_setting[3][1] +
                         self.inverted_residual_setting[4][1] +
                         self.inverted_residual_setting[5][1] +
                         self.inverted_residual_setting[6][1])

        feat_chns = int(org_last_chns * width_multi) if width_multi > 1.0 else org_last_chns
        self.feat_fuse = InvertedResidualIABN(inp=feat_chns, oup=fuse_chns, stride=1, dilate=1,
                                              expand_ratio=1, norm_act=norm_act)

        self.pyramid_pool = DSASPPInPlaceABNBlock(fuse_chns, aspp_chns, aspp_dilate=aspp_dilate, norm_act=norm_act)

        feat_chns = aspp_chns + self.inverted_residual_setting[1][1] + self.inverted_residual_setting[0][1]
        self.final_fuse = InvertedResidualIABN(inp=feat_chns, oup=aspp_chns, stride=1, dilate=1,
                                               expand_ratio=1, norm_act=norm_act)

        self.aspp_scse = PBCSABlock(in_chns=aspp_chns, reduct_ratio=16, dilation=16, use_res=True)
        self.score = nn.Sequential(OrderedDict([("dropout", nn.Dropout2d(0.175)),
                                                ("norm", norm_act(aspp_chns)),
                                                ("conv", nn.Conv2d(aspp_chns, self.num_classes,
                                                                   kernel_size=1, stride=1,
                                                                   padding=0, bias=True))]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.10, mode='fan_in', nonlinearity='leaky_relu')
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
        super(MobileNetV2Plus, self).train()
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
        assert (in_h % 8 == 0 and in_w % 8 == 0), "> in_size must product of 8!!!"

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(x)     # (N, 32,   H/2, W/2)  1/2
        stg1 = self.mod2(stg1)  # (N, 16,   H/2, W/2)  1/2 -> 1/4 -> 1/8
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)  # 1/4
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)  # 1/8

        stg2 = self.mod3(stg1)  # (N, 24,   H/4, W/4)  1/4 -> 1/8
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, padding=1)  # 1/8

        stg3 = self.mod4(stg2)  # (N, 32,   H/8, W/8)  1/8
        stg4 = self.mod5(stg3)  # (N, 64,   H/8, W/8)  1/8 dilation=2
        stg5 = self.mod6(stg4)  # (N, 96,   H/8, W/8)  1/8 dilation=4
        stg6 = self.mod7(stg5)  # (N, 160,  H/8, W/8)  1/8 dilation=8
        stg7 = self.mod8(stg6)  # (N, 320,  H/8, W/8)  1/8 dilation=16

        # (N, 712, 56,  112)  1/8  (16+24+32+64+96+160+320)
        feat = self.feat_fuse(torch.cat((stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1), dim=1))

        # dsn = None
        # if self.training:
        #     dsn = self.dsn(feat)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # (N, 256+24+16=296, H/4, W/4)
        feat = self.aspp_scse(self.final_fuse(torch.cat((stg2, self.pyramid_pool(feat), stg1_1), dim=1)))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier: pixel-wise classification-segmentation
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        feat = F.interpolate(input=self.score(feat), size=(in_h, in_w), mode="bilinear", align_corners=True)
        return feat


if __name__ == '__main__':
    import time
    import torch
    # from utils.utils import freeze_bn

    model = MobileNetV2Plus(num_classes=19,
                            width_multi=1.0,
                            fuse_chns=512,
                            aspp_chns=256,
                            aspp_dilate=(12, 24, 36),
                            norm_act=InPlaceABNSync)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    model.eval()
    with torch.no_grad():
        while True:
            dummy_in = torch.randn(2, 3, 768, 768).cuda()
            start_time = time.time()
            dummy_out = model(dummy_in)
            del dummy_out
            print("> Inference Time: {}".format(time.time() - start_time))
