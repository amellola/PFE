"""Specified default transforms."""
# pylint: disable=unused-import
import logging
from typing import Callable, Sequence, Union

import numpy as np

from .additional import CleanTransform, PrintD, PrintFilenameD

# pylint: disable-next=wrong-import-order
from monai.transforms import (  # isort:skip
    EnsureChannelFirstD,
    Identity,
    LoadImageD,
    ResizeD,
    ToTensorD,
    TransposeD,
    CenterSpatialCropD,
    SpatialPadD,
    DataStats,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ToTensor,
    Transpose,
    CenterSpatialCrop,
    SpatialPad,
    ConcatItemsD,
    DeleteItemsD,
    NormalizeIntensityD,
    AsDiscreted,
    SaveImageD,
    EnsureTyped,
    Activationsd,
    Invertd,
    RandCropByPosNegLabeld,
    RandFlipd,

)

log = logging.getLogger(__name__)


class Segmentation:
    """Holds all transforms for segmentation."""

    def __init__(
        self,
        resize,
        device,
        cm_channel,
        cm_loss,
        size,
    ):
        """Initialize the transformer."""
        self.resize = resize
        self.device = device
        self.cm_channel = cm_channel
        self.cm_loss = cm_loss
        self.size = size

    def default(self):
        """Specify default transforms."""
        return [
            LoadImageD(keys=("img", "seg"), image_only=True),
            ToTensorD(keys=("img", "seg")),
            TransposeD(keys=("img", "seg"), indices=(2, 1, 0)),
            EnsureChannelFirstD(keys=("img", "seg")),
            ResizeD(
                keys=("img", "seg"),
                spatial_size=(self.resize, self.resize, self.resize),
                mode=["trilinear", "nearest"],
                align_corners=[False, None],
            )
            if self.resize is not None
            else Identity(),
        ]

    def transforms(self):
        """Specify transforms for training."""
        # h, w = self.size, self.size
        # spatial_size = (int(3.125 * resize), resize, resize)
        spatial_size = (self.resize, self.resize, self.resize)
        axis_order = (2, 1, 0)
        return [
            # PrintD(keys=("img",)),
            LoadImageD(
                keys=("img", "seg", "cm"), image_only=True, allow_missing_keys=True
            ),
            ToTensorD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            # PrintFilenameD(keys=("img",)),
            TransposeD(
                keys=("img", "seg", "cm"), indices=axis_order, allow_missing_keys=True
            ),
            # CleanTransformD(keys=("seg",), only_sol=False),
            EnsureChannelFirstD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            # CenterSpatialCropD(
            #     keys=("img", "seg", "cm"), roi_size=(-1, h, w), allow_missing_keys=True
            # ),
            # SpatialPadD(
            #     keys=("img", "seg", "cm"), spatial_size=(-1, h, w), allow_missing_keys=True
            # ),
            ResizeD(
                keys=("img", "seg", "cm"),
                spatial_size=spatial_size,
                mode=["trilinear", "nearest", "trilinear"],
                align_corners=[False, None, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            RandFlipd(keys=["img","seg"], spatial_axis=0, prob=0.5),
            #NormalizeIntensityD(keys=("img",)),
            # === Confidence Maps ===
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
        ]

    def pre_transforms(self) -> Sequence[Callable]:
        """Pre inference transforms."""
        # h, w = self.size, self.size
        axis_order = (2, 1, 0)
        spatial_size = (self.resize, self.resize, self.resize)
        return [
            LoadImageD(
                keys=("img", "seg", "cm"), image_only=True, allow_missing_keys=True
            ),
            ToTensorD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            TransposeD(
                keys=("img", "seg", "cm"), indices=axis_order, allow_missing_keys=True
            ),
            EnsureChannelFirstD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            # CenterSpatialCropD(
            #     keys=("img", "seg", "cm"), roi_size=(-1, h, w), allow_missing_keys=True
            # ),
            # SpatialPadD(
            #     keys=("img", "seg", "cm"),
            #     spatial_size=(-1, h, w),
            #     allow_missing_keys=True,
            # ),
            ResizeD(
                keys=("img", "cm"),
                spatial_size=spatial_size,
                mode=["trilinear", "trilinear"],
                align_corners=[False, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            # === Confidence Maps ===
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0, allow_missing_keys=True),
            RandFlipd(keys=["img","seg"], spatial_axis=0, prob=0.5)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
        ]

    def inverse_transforms(self, transforms_to_inverse):
        """Specify inverse transforms for post-processing."""
        return [
            EnsureTyped(keys=["pred", "seg"], allow_missing_keys=True),
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys=("pred",),
                transform=transforms_to_inverse,
                orig_keys=("img",),
                meta_keys=("pred_meta_dict",),
                orig_meta_keys=("img_meta_dict",),
                meta_key_postfix="meta_dict",
                to_tensor=True,
                nearest_interp=False,
            ),
        ]

    def transforms_to_inverse(self) -> Union[None, Sequence[Callable]]:
        """Specify inverse transforms for post-processing.

        Provide List of inverse-transforms.  They are normally subset of pre-transforms.
        This task is performed on output_label (using the references from input_key)
        :param data: current data dictionary/request which can be helpful to define the transforms per-request basis
        Return one of the following.
            - None: Return None to disable running any inverse transforms (default behavior).
            - Empty: Return [] to run all applicable pre-transforms which has inverse method
            - list: Return list of specific pre-transforms names/classes to run inverse method
            For Example::
                return [
                    monai.transforms.Spacingd,
                ]
        """
        return [ResizeD]

    def post_transforms(self, output_dir, save=True) -> Sequence[Callable]:
        """Post inference transforms."""
        return [
            TransposeD(
                keys=("pred", "seg"), indices=(0, 3, 2, 1)
            ),  # transform back to original shape (not invertible as of 4th dimension)
            AsDiscreted(keys=("pred",), argmax=True, to_onehot=None),
            SaveImageD(
                keys=("pred",),
                meta_keys=("pred_meta_dict",),
                output_dir=output_dir,
                output_postfix="seg",
                squeeze_end_dims=True,
                separate_folder=False,
                output_dtype=np.uint8,
            )
            if save
            else Identity(),
            AsDiscreted(keys=("pred", "seg"), to_onehot=7),
        ]


def transform_single(_resize, _mode, _align_corners, skip_label_transform=False):
    """Specify default transforms for single image."""
    h, w = 512, 512
    # spatial_size = (int(3.125 * resize), resize, resize)
    # spatial_size = (resize, resize, resize)
    axis_order = (2, 1, 0)
    return [
        LoadImage(image_only=True),
        ToTensor(),
        Transpose(indices=axis_order),
        EnsureChannelFirst(),
        CleanTransform(only_sol=False, skip=skip_label_transform),
        CenterSpatialCrop((-1, h, w)),
        SpatialPad((-1, h, w)),
        # Resize(
        #     spatial_size=spatial_size,
        #     mode=mode,
        #     align_corners=align_corners,
        # )
        # if resize is not None
        # else Identity(),
        DataStats(),
    ]
