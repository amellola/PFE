import numpy as np
import torch
from monai.losses import DiceLoss, FocalLoss
from torch.nn import CrossEntropyLoss


class Segmentation:
    """Config for DeepAtlas Losses."""

    def __init__(self, num_segmentation_classes, device):
        self.num_segmentation_classes = num_segmentation_classes
        if num_segmentation_classes == 7:
            weight = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        elif num_segmentation_classes == 2:
            weight = torch.Tensor([0.2, 0.8])
        elif num_segmentation_classes == 1:
            weight = torch.Tensor([1])
        self._xent_loss = CrossEntropyLoss(weight=weight.to(device))
        self._dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
            softmax=True,  # Note that our segmentation network is missing the softmax at the end
            reduction="mean",
        )
        self._focal_loss = FocalLoss(
            include_background=True,
            to_onehot_y=True,
            reduction="mean",
        )
        self.device = device

    def ce_loss(self, pred, gt, cm=None):
        """Compute the confidence map loss."""
        # calc one hot for all channels in gt_onehot
        shape = list(gt.shape)
        shape[1] = self.num_segmentation_classes
        gt_onehot = np.zeros(shape)
        if len(shape) == 5:
            gt_onehot[:, [0], :, :, :] = (gt == 0).astype(float)
        elif len(shape) == 4:
            gt_onehot[:, [0], :, :] = (gt == 0).astype(float)
        elif len(shape) == 3:
            gt_onehot[:, [0], :] = (gt == 0).astype(float)
        for i in range(1, self.num_segmentation_classes):
            if cm is not None:
                if len(shape) == 5:
                    gt_onehot[:, [i], :, :, :] = (gt == i).astype(float) * cm[
                        :, [0], :, :, :
                    ]
                elif len(shape) == 4:
                    gt_onehot[:, [i], :, :] = (gt == i).astype(float) * cm[:, [0], :, :]
                elif len(shape) == 3:
                    gt_onehot[:, [i], :] = (gt == i).astype(float) * cm[:, [0], :]
            else:
                if len(shape) == 5:
                    gt_onehot[:, [i], :, :, :] = (gt == i).astype(float)
                elif len(shape) == 4:
                    gt_onehot[:, [i], :, :] = (gt == i).astype(float)
                elif len(shape) == 3:
                    gt_onehot[:, [i], :] = (gt == i).astype(float)

        # compute the cross-entropy loss
        return self._xent_loss(pred, torch.from_numpy(gt_onehot).to(pred.device))

    def dice_ce_loss(
        self,
        pred,
        gt,
        cm=None,
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.5,
    ):
        dice_loss = self.dice_loss(pred, gt)
        ce_loss = self.ce_loss(pred, gt, cm=cm)
        total_loss: torch.Tensor = lambda_dice * dice_loss + lambda_ce * ce_loss

        return total_loss

    def dice_loss(self, pred, gt, cm=None):  # pylint: disable=unused-argument
        return self._dice_loss(pred, gt)

    def focal_loss(self, pred, gt, cm=None):
        return self._focal_loss(pred, gt)
