# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation
# ---------------------------------------------------------------------------------------------------------------- #
# Compute Metrics for Semantic Segmentation
# class:
#       > AverageMeter
#       > RunningMetrics
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   10.10.2018
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningMetrics(object):
    def __init__(self, num_classes):
        """
        Computes and stores the Metric values from Confusion Matrix
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc

        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix

        :param num_classes: <int> number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix

        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < self.num_classes)
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        """
        Compute Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix

        :param label_gts: <np.ndarray> ground-truths
        :param label_preds: <np.ndarray> predictions
        :return:
        """
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """
        Reset Confusion Matrix
        :return:
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_scores(self):
        """
        Returns score about:
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc

        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix

        :return:
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)

        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        acc_cls = tp / (sum_a1 + np.finfo(np.float32).eps)
        acc_cls = np.nanmean(acc_cls)

        # ---------------------------------------------------------------------- #
        # 2. Frequency weighted Accuracy & Mean IoU
        # ---------------------------------------------------------------------- #
        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.num_classes), iu))

        return {'Overall_Acc': acc,
                'Mean_Acc': acc_cls,
                'FreqW_Acc': fwavacc,
                'Mean_IoU': mean_iu}, cls_iu


# ---------------------------------------------- #
# Test code
# ---------------------------------------------- #
if __name__ == "__main__":
    score = RunningMetrics(2)

    gt = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])

    score.update(gt, pred)
    print(score.confusion_matrix)
