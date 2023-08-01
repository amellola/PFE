import monai
import numpy as np
import torch
from monai.losses import (
    BendingEnergyLoss,
    DiceLoss,
    LocalNormalizedCrossCorrelationLoss,
)
from torch.nn import CrossEntropyLoss


class Deepatlas:
    """Config for DeepAtlas Losses."""

    def __init__(self, num_segmentation_classes, loss="dice"):
        self.num_segmentation_classes = num_segmentation_classes
        self._dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
            softmax=True,  # Note that our segmentation network is missing the softmax at the end
            reduction="mean",
        )
        self._dice_loss2 = DiceLoss(
            include_background=True, to_onehot_y=False, softmax=False, reduction="mean"
        )
        self._lncc_loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3, kernel_size=3, kernel_type="rectangular", reduction="mean"
        )
        self._bending_loss = BendingEnergyLoss()

        # warp for using displacement field from regisration network to warp segmentation
        self.warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
        self.warp_nearest = monai.networks.blocks.Warp(
            mode="nearest", padding_mode="border"
        )

        self._ce_loss = CrossEntropyLoss()

        # Added for unetr
        self._diceCE_loss = monai.losses.DiceCELoss(
            include_background=True,
            to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
            softmax=True,  # Note that our segmentation network is missing the softmax at the end
        )

        if loss == "dice":
            self.seg_train_loss = self._dice_loss2_wrapper
        elif loss == "dice_ce":
            self.seg_train_loss = self._dice_ce_loss
        elif loss == "cross_entropy":
            self.seg_train_loss = self._ce_loss_wrapper

    def dice_loss(self, *args, **kwargs):
        return self._dice_loss(*args, **kwargs)

    def _ce_loss_wrapper(self, pred, gt, cm=None):
        """Compute the confidence map loss."""
        # calc one hot for all channels in gt_onehot
        for i in range(1, self.num_segmentation_classes):
            if cm is not None:
                gt[:, [i], :, :, :] = gt[:, [i], :, :, :] * cm[:, [0], :, :, :]

        # compute the cross-entropy loss
        return self._ce_loss(pred, gt)

    def _dice_ce_loss(self, pred, gt, cm=None, lambda_dice=0.5, lambda_ce=0.5):
        dice_loss = self._dice_loss2_wrapper(pred, gt)
        ce_loss = self._ce_loss_wrapper(pred, gt, cm=cm)
        total_loss: torch.Tensor = lambda_dice * dice_loss + lambda_ce * ce_loss

        return total_loss

    def _dice_loss2_wrapper(self, pred, gt, cm=None):
        # A version of the dice loss with to_onehot_y=False and softmax=False;
        # This will be handy for anatomy loss, for which we often compare two outputs of seg_net
        return self._dice_loss2(pred, gt)

    def lncc_loss(self, *args, **kwargs):
        return self._lncc_loss(*args, **kwargs)

    def bending_loss(self, *args, **kwargs):
        return self._bending_loss(*args, **kwargs)

    def regularization_loss(self, *args, **kwargs):
        return self._bending_loss(*args, **kwargs)

    def similarity_loss(self, displacement_field, image_pair, cm_channel=False):
        """Similarity loss for joint registration and segmentation.

        Accepts a batch of displacement fields, shape (B,3,H,W,D),
        and a batch of image pairs, shape (B,2,H,W,D).
        """
        warped_img2 = self.warp(
            image_pair[:, [2 if cm_channel else 1], :, :, :], displacement_field
        )
        return self.lncc_loss(
            warped_img2, image_pair[:, [0], :, :, :]  # prediction  # target
        )

    def anatomy_loss(
        self,
        displacement_field,
        image_pair,
        seg_net,
        gt_seg1=None,
        gt_seg2=None,
        cm_channel=False,
    ):
        """Anatomy loss for joint registration and segmentation.

        Accepts a batch of displacement fields, shape (B,3,H,W,D),
        and a batch of image pairs, shape (B,2,H,W,D).
        seg_net is the model used to segment an image,
        mapping (B,1,H,W,D) to (B,C,H,W,D) where C is the number of segmentation classes.
        gt_seg1 and gt_seg2 are ground truth segmentations for the images in image_pair, if ground truth is available;
        if unavailable then they can be None.
        gt_seg1 and gt_seg2 are expected to be in the form of class labels, with shape (B,1,H,W,D).
        """
        if gt_seg1 is not None:
            # ground truth seg of target image
            seg1 = monai.networks.one_hot(gt_seg1, self.num_segmentation_classes)
        else:
            # seg_net on target image, "noisy ground truth"
            if cm_channel:
                seg1 = seg_net(image_pair[:, :2, :, :, :]).softmax(dim=1)
            else:
                seg1 = seg_net(image_pair[:, [0], :, :, :]).softmax(dim=1)

        if gt_seg2 is not None:
            # ground truth seg of moving image
            seg2 = monai.networks.one_hot(gt_seg2, self.num_segmentation_classes)
        else:
            # seg_net on moving image, "noisy ground truth"
            if cm_channel:
                seg2 = seg_net(image_pair[:, 2:, :, :, :]).softmax(dim=1)
            else:
                seg2 = seg_net(image_pair[:, [1], :, :, :]).softmax(dim=1)

        # seg1 and seg2 are now in the form of class probabilities at each voxel
        # The trilinear interpolation of the function `warp` is then safe to use;
        # it will preserve the probabilistic interpretation of seg2.

        return self.seg_train_loss(
            self.warp(seg2, displacement_field),  # warp of moving image segmentation
            seg1,  # target image segmentation
        )

    # Function for forward pass of reg_net, to avoid duplicating code between training and validation
    def reg_losses(self, batch, device, reg_net, seg_net, cm_channel=False):
        """Compute registration losses for a batch of data."""
        img12 = batch["img12"].to(device)

        displacement_field12 = reg_net(img12)

        loss_sim = self.similarity_loss(
            displacement_field12, img12, cm_channel=cm_channel
        )

        loss_reg = self.regularization_loss(displacement_field12)

        gt_seg1 = batch["seg1"].to(device) if "seg1" in batch.keys() else None
        gt_seg2 = batch["seg2"].to(device) if "seg2" in batch.keys() else None
        loss_ana = self.anatomy_loss(
            displacement_field12,
            img12,
            seg_net,
            gt_seg1,
            gt_seg2,
            cm_channel=cm_channel,
        )

        return loss_sim, loss_reg, loss_ana
