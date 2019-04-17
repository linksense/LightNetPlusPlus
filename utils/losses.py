# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# Compute Metrics for Semantic Segmentation
# class:
#       > BootstrappedCrossEntropy2D
#       > DiceLoss2D
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   10.10.2018
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
import torch.nn.functional as F
import scipy.ndimage as nd
import torch.nn as nn
import numpy as np
import torch


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Bootstrapped CrossEntropy2D
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class BootstrappedCrossEntropy2D(nn.Module):
    def __init__(self, top_k=128, ignore_index=-100):
        """
        Bootstrapped CrossEntropy2D: The pixel-bootstrapped cross entropy loss

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(BootstrappedCrossEntropy2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                                         5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
                                         5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
                                         2.78376194]).cuda()
        self.top_k = top_k
        self.ignore_index = ignore_index

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :param top_k: <int> Top-K worst predictions
        :return: <torch.Tensor> loss
        """
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()

                # ------------------------------------------------------------------------ #
                # 1. Compute CrossEntropy Loss without Reduction
                # ------------------------------------------------------------------------ #
                # target_mask = (targets >= 0) * (targets != self.ignore_index)
                # targets = targets[target_mask]

                batch_loss = F.cross_entropy(input=predict, target=targets,
                                             weight=None,
                                             ignore_index=self.ignore_index,  reduction='none')

                # ------------------------------------------------------------------------ #
                # 2. Bootstrap from each image not entire batch
                #    For each element in the batch, collect the top K worst predictions
                # ------------------------------------------------------------------------ #
                loss = 0.0

                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h*feat_w)

                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k

                loss_fuse += loss / float(batch_size)
        else:
            batch_size, channels, feat_h, feat_w = predictions.size()

            # ------------------------------------------------------------------------ #
            # 1. Compute CrossEntropy Loss without Reduction
            # ------------------------------------------------------------------------ #
            # target_mask = (targets >= 0) * (targets != self.ignore_index)
            # targets = targets[target_mask]

            batch_loss = F.cross_entropy(input=predictions, target=targets,
                                         weight=None,
                                         ignore_index=self.ignore_index, reduction='none')

            # ------------------------------------------------------------------------ #
            # 2. Bootstrap from each image not entire batch
            #    For each element in the batch, collect the top K worst predictions
            # ------------------------------------------------------------------------ #
            loss = 0.0
            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)

                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k

            loss_fuse += loss / float(batch_size)

        return loss_fuse


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, top_k=512*512, ignore_index=255):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = BootstrappedCrossEntropy2D(top_k=top_k, ignore_index=ignore_index)

    def _update_topk(self, top_k):
        self.criterion._update_topk(top_k)

    def forward(self, predictions, targets):
        loss1 = self.criterion(predictions[0], targets)
        loss2 = self.criterion(predictions[1], targets)

        return loss1 + loss2 * 0.4


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Bootstrapped CrossEntropy2D
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class OHEMBootstrappedCrossEntropy2D(nn.Module):
    def __init__(self, factor=8.0, thresh=0.7, min_kept=100000, top_k=128, ignore_index=-100):
        """
        Bootstrapped CrossEntropy2D: The pixel-bootstrapped cross entropy loss

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(OHEMBootstrappedCrossEntropy2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                                         5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
                                         5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
                                         2.78376194])
        self.top_k = top_k
        self.ignore_index = ignore_index

        self.factor = factor
        self.thresh = thresh
        self.min_kept = int(min_kept)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        threshold = 1.0
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh

            if min_kept > 0:
                k_th = min(len(pred), int(min_kept)) - 1
                new_array = np.partition(pred, int(k_th))
                new_threshold = new_array[k_th]

                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :param top_k: <int> Top-K worst predictions
        :return: <torch.Tensor> loss
        """
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()

                # ------------------------------------------------------------------------ #
                # 1. Compute CrossEntropy Loss without Reduction
                # ------------------------------------------------------------------------ #
                # target_mask = (targets >= 0) * (targets != self.ignore_index)
                # targets = targets[target_mask]

                input_prob = F.softmax(predict, dim=1)
                targets = self.generate_new_target(input_prob, targets)
                batch_loss = F.cross_entropy(input=predict, target=targets,
                                             weight=self.weight.cuda(predictions.get_device()),
                                             ignore_index=self.ignore_index,  reduction='none')

                # ------------------------------------------------------------------------ #
                # 2. Bootstrap from each image not entire batch
                #    For each element in the batch, collect the top K worst predictions
                # ------------------------------------------------------------------------ #
                loss = 0.0

                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h*feat_w)

                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k

                loss_fuse += loss / float(batch_size)
        else:
            batch_size, channels, feat_h, feat_w = predictions.size()

            # ------------------------------------------------------------------------ #
            # 1. Compute CrossEntropy Loss without Reduction
            # ------------------------------------------------------------------------ #
            # target_mask = (targets >= 0) * (targets != self.ignore_index)
            # targets = targets[target_mask]
            input_prob = F.softmax(predictions, dim=1)
            targets = self.generate_new_target(input_prob, targets)

            batch_loss = F.cross_entropy(input=predictions, target=targets,
                                         weight=self.weight.cuda(predictions.get_device()),
                                         ignore_index=self.ignore_index, reduction='none')

            # ------------------------------------------------------------------------ #
            # 2. Bootstrap from each image not entire batch
            #    For each element in the batch, collect the top K worst predictions
            # ------------------------------------------------------------------------ #
            loss = 0.0

            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)

                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k

            loss_fuse += loss / float(batch_size)

        return loss_fuse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Focal Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """
    def __init__(self, top_k=128, ignore_label=250, alpha=0.25, gamma=2):
        """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        :param ignore_label:  <int> ignore label
        :param alpha:         <torch.Tensor> the scalar factor
        :param gamma:         <float> gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        """
        super(FocalLoss2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                                         5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
                                         5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
                                         2.78376194])

        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k

        self.ignore_label = ignore_label
        self.one_hot = torch.eye(self.num_classes)

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        assert not targets.requires_grad

        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()

                # ------------------------------------------------------------------------ #
                # 1. Compute CrossEntropy Loss without Reduction
                # ------------------------------------------------------------------------ #
                batch_loss = F.cross_entropy(input=predict, target=targets,
                                             weight=self.weight.cuda(predictions.get_device()),
                                             ignore_index=self.ignore_index, reduction='none')

                # ------------------------------------------------------------------------ #
                # 2. Bootstrap from each image not entire batch
                #    For each element in the batch, collect the top K worst predictions
                # ------------------------------------------------------------------------ #
                loss = 0.0

                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h * feat_w)

                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k

                loss_fuse += loss / float(batch_size)

        else:
            batch_size, channels, feat_h, feat_w = predictions.size()

            # ------------------------------------------------------------------------ #
            # 1. Compute CrossEntropy Loss without Reduction
            # ------------------------------------------------------------------------ #

            batch_loss = F.cross_entropy(input=predictions, target=targets,
                                         weight=self.weight.cuda(predictions.get_device()),
                                         ignore_index=self.ignore_index, reduction='none')

            # ------------------------------------------------------------------------ #
            # 2. Bootstrap from each image not entire batch
            #    For each element in the batch, collect the top K worst predictions
            # ------------------------------------------------------------------------ #
            loss = 0.0

            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)

                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k

            loss_fuse += loss / float(batch_size)

        log_pt = -loss_fuse
        return -((1.0 - torch.exp(log_pt)) ** self.gamma) * self.alpha * log_pt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Semantic Encoding Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SemanticEncodingLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_label=250, weight=None, alpha=0.25):
        """
        Semantic Encoding Loss
        :param num_classes: <int> Number of classes
        :param ignore_label: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param alpha: <float> A manual rescaling weight given to Semantic Encoding Loss
        """
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha

        self.num_classes = num_classes
        self.weight = weight
        self.ignore_label = ignore_label

    def __unique_encode(self, msk_targets):
        """

        :param cls_targets: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :return:
        """
        batch_size, _, _ = msk_targets.size()
        target_mask = (msk_targets >= 0) * (msk_targets != self.ignore_label)
        cls_targets = [msk_targets[idx].masked_select(target_mask[idx]) for idx in np.arange(batch_size)]

        # unique_cls = [np.unique(label.numpy(), return_counts=True) for label in cls_targets]
        unique_cls = [torch.unique(label) for label in cls_targets]

        encode = torch.zeros(batch_size, self.num_classes, dtype=torch.float32, requires_grad=False)

        for idx in np.arange(batch_size):
            index = unique_cls[idx].long()
            encode[idx].index_fill_(dim=0, index=index, value=1.0)

        return encode

    def forward(self, predictions, targets):
        """
        
        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return:
        """
        enc_targets = self.__unique_encode(targets)
        se_loss = F.binary_cross_entropy_with_logits(predictions, enc_targets, weight=self.weight,
                                                     reduction="elementwise_mean")

        return self.alpha * se_loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Dice Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class DiceLoss2D(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        """
        Dice Loss for Semantic Segmentation

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(DiceLoss2D, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1.0e-6

        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:

                predict = F.softmax(predict, dim=1)
                encoded_target = predict.detach() * 0  # The result will never require gradient.

                # ----------------------------------------------------- #
                # 1. Targets Pre-processing & Encoding
                # ----------------------------------------------------- #
                mask = None
                if self.ignore_index is not None:
                    mask = targets == self.ignore_index
                    targets = targets.clone()
                    targets[mask] = 0

                    encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)
                    mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
                    encoded_target[mask] = 0
                else:
                    encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)

                if self.weight is None:
                    self.weight = 1.0

                # ----------------------------------------------------- #
                # 2. Compute Dice Coefficient
                # ----------------------------------------------------- #
                intersection = predictions * encoded_target
                denominator = predictions + encoded_target

                if self.ignore_index is not None:
                    denominator[mask] = 0

                numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
                denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth

                # ----------------------------------------------------- #
                # 3. Compute Weighted Dice Loss
                # ----------------------------------------------------- #
                loss_per_channel = self.weight * (1.0 - (numerator / denominator))

                loss_fuse = loss_per_channel.sum() / predictions.size(1)
        else:
            predict = F.softmax(predictions, dim=1)
            encoded_target = predict.detach() * 0  # The result will never require gradient.

            # ----------------------------------------------------- #
            # 1. Targets Pre-processing & Encoding
            # ----------------------------------------------------- #
            mask = None
            if self.ignore_index is not None:
                mask = targets == self.ignore_index
                targets = targets.clone()
                targets[mask] = 0

                encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)
                mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)

            if self.weight is None:
                self.weight = 1.0

            # ----------------------------------------------------- #
            # 2. Compute Dice Coefficient
            # ----------------------------------------------------- #
            intersection = predictions * encoded_target
            denominator = predictions + encoded_target

            if self.ignore_index is not None:
                denominator[mask] = 0

            numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
            denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth

            # ----------------------------------------------------- #
            # 3. Compute Weighted Dice Loss
            # ----------------------------------------------------- #
            loss_per_channel = self.weight * (1.0 - (numerator / denominator))

            loss_fuse = loss_per_channel.sum() / predictions.size(1)

        return loss_fuse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Soft-Jaccard Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SoftJaccardLoss2D(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        """
        Soft-Jaccard Loss for Semantic Segmentation

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(SoftJaccardLoss2D, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1.0

        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0  # The result will never require gradient.

        # ----------------------------------------------------- #
        # 1. Targets Pre-processing & Encoding
        # ----------------------------------------------------- #
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0

            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)

        if self.weight is None:
            self.weight = 1.0

        # ----------------------------------------------------- #
        # 2. Compute Jaccard Coefficient
        # ----------------------------------------------------- #
        intersection = predictions * encoded_target
        denominator = predictions + encoded_target

        if self.ignore_index is not None:
            denominator[mask] = 0

        numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1)
        denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1)

        # ----------------------------------------------------- #
        # 3. Compute Weighted Soft-Jaccard Loss
        # ----------------------------------------------------- #
        # loss_per_channel = self.weight * (1.0 - torch.log(((numerator + smooth) / (denominator - numerator + smooth))))
        loss_per_channel = self.weight * (1.0 - ((numerator + smooth) / (denominator - numerator + smooth)))

        return loss_per_channel.sum() / predictions.size(1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Tversky Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class TverskyLoss2D(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, weight=None, ignore_index=-100):
        """
        Tversky Loss for Semantic Segmentation

        :param alpha: <int> Parameter to control precision and recall
        :param beta: <int> Parameter to control precision and recall
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(TverskyLoss2D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1.0

        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0  # The result will never require gradient.

        # ----------------------------------------------------- #
        # 1. Targets Pre-processing & Encoding
        # ----------------------------------------------------- #
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0

            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)

        if self.weight is None:
            self.weight = 1.0

        # ----------------------------------------------------- #
        # 2. Compute Tversky Index
        # ----------------------------------------------------- #
        intersection = predictions * encoded_target
        numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth

        ones = torch.ones_like(predictions)
        item1 = predictions * (ones - encoded_target)
        item2 = (ones - predictions) * encoded_target
        denominator = numerator + self.alpha * item1.sum(dim=0).sum(dim=1).sum(dim=1) + \
                      self.beta * item2.sum(dim=0).sum(dim=1).sum(dim=1)

        if self.ignore_index is not None:
            denominator[mask] = 0

        # ----------------------------------------------------- #
        # 3. Compute Weighted Tversky Loss
        # ----------------------------------------------------- #
        # loss_per_channel = self.weight * (1.0 - torch.log(((numerator) / (denominator - numerator))))
        loss_per_channel = self.weight * (1.0 - (numerator / (denominator - numerator)))

        return loss_per_channel.sum() / predictions.size(1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Asymmetric Similarity Loss Function to Balance Precision and Recall in
# Highly Unbalanced Deep Medical Image Segmentation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class AsymmetricSimilarityLoss2D(nn.Module):
    def __init__(self, beta=0.6, weight=None, ignore_index=-100):
        """
        Tversky Loss for Semantic Segmentation

        :param beta: <int> Parameter to control precision and recall
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(AsymmetricSimilarityLoss2D, self).__init__()
        self.beta = beta
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        eps = 1e-8
        beta = self.beta ** 2

        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0  # The result will never require gradient.

        # ----------------------------------------------------- #
        # 1. Targets Pre-processing & Encoding
        # ----------------------------------------------------- #
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0

            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1), value=1.0)

        if self.weight is None:
            self.weight = 1.0

        # ----------------------------------------------------- #
        # 2. Compute F-beta score
        # ----------------------------------------------------- #
        intersection = predictions * encoded_target
        numerator = (1.0 + beta) * intersection.sum(dim=0).sum(dim=1).sum(dim=1)

        ones = torch.ones_like(predictions)
        item1 = predictions * (ones - encoded_target)
        item2 = (ones - predictions) * encoded_target
        denominator = numerator + beta * item1.sum(dim=0).sum(dim=1).sum(dim=1) + \
                      item2.sum(dim=0).sum(dim=1).sum(dim=1) + eps

        if self.ignore_index is not None:
            denominator[mask] = 0

        # ----------------------------------------------------- #
        # 3. Compute Weighted Tversky Loss
        # ----------------------------------------------------- #
        # loss_per_channel = self.weight * (1.0 - torch.log(((numerator) / (denominator - numerator))))
        loss_per_channel = self.weight * (1.0 - (numerator / (denominator - numerator)))

        return loss_per_channel.sum() / predictions.size(1)


# ---------------------------------------------- #
# Test code
# ---------------------------------------------- #
if __name__ == "__main__":
    dummy_in = torch.LongTensor(32, 19).random_(0, 19).requires_grad_()
    dummy_gt = torch.LongTensor(32, 32, 32).random_(0, 19)

    se_loss = SemanticEncodingLoss(num_classes=19, ignore_label=250, weight=None, alpha=0.25)

    while True:
        top_k = 256
        loss = se_loss(dummy_in, dummy_gt)

        print("Loss: {}".format(loss.item()))
