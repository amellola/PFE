"""Specified default transforms."""
# pylint: disable=unused-import
import logging
from typing import Callable, Sequence, Union

import numpy as np
from monai.data import PILReader

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
    RandFlipD,
    RandAffined,
    RandRotated,
    RandGaussianNoiseD,
    EnsureTyped,
    Activationsd,
    Invertd,
    Lambdad,
    ScaleIntensityd,
)


class Segmentation2D:
    """Hold all transforms for 2D segmentation."""

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

    def transforms(self):
        """Specify transforms for training."""
        spatial_size = (self.resize, self.resize)
        return [
            LoadImageD(
                keys=("img", "seg", "cm"),
                image_only=True,
                allow_missing_keys=True,
                reader=PILReader(converter=lambda image: image.convert("L")),
            ),
            EnsureChannelFirstD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            TransposeD(
                keys=("img", "seg", "cm"), indices=(0, 2, 1), allow_missing_keys=True
            ),
            RandFlipD(
                keys=("img", "seg", "cm"),
                spatial_axis=1,
                prob=0.5,
                allow_missing_keys=True,
            ),
            RandFlipD(
                keys=("img", "seg", "cm"),
                spatial_axis=0,
                prob=0.5,
                allow_missing_keys=True,
            ),
            ToTensorD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            NormalizeIntensityD(keys=("img")),
            # NormalizeIntensityD(keys=("seg"), divisor=255, subtrahend=0, dtype=np.int8),
            RandGaussianNoiseD(keys=("img",), prob=0.5, std=0.1),
            ResizeD(
                keys=("img", "seg", "cm"),
                spatial_size=spatial_size,
                mode=["bilinear", "nearest", "bilinear"],
                align_corners=[False, None, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            # === Confidence Maps ===
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
        ]

    def pre_transforms(self) -> Sequence[Callable]:
        """Pre inference transforms."""
        spatial_size = (self.resize, self.resize)
        return [
            LoadImageD(
                keys=("img", "seg", "cm"),
                image_only=True,
                allow_missing_keys=True,
                reader=PILReader(converter=lambda image: image.convert("L")),
            ),
            EnsureChannelFirstD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            TransposeD(keys=("img", "cm"), indices=(0, 2, 1), allow_missing_keys=True),
            ToTensorD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            NormalizeIntensityD(keys=("img")),
            # NormalizeIntensityD(keys=("seg"), divisor=255, subtrahend=0, dtype=np.int8),
            ResizeD(
                keys=("img", "cm"),
                spatial_size=spatial_size,
                mode=["bilinear", "bilinear"],
                align_corners=[False, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            # === Confidence Maps ===
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
        ]

    def inverse_transforms(self, transforms_to_inverse):
        """Specify inverse transforms for post-processing."""
        return [
            EnsureTyped(keys=["pred", "seg"], allow_missing_keys=True),
            # Activationsd(keys="pred", softmax=True),
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
            TransposeD(keys=("pred",), indices=(0, 2, 1)),
            AsDiscreted(keys=("pred",), argmax=True, to_onehot=None),
            SaveImageD(
                keys=("pred",),
                meta_keys=("pred_meta_dict",),
                output_dir=output_dir,
                output_postfix="seg",
                resample=True,
                mode="nearest",
                squeeze_end_dims=True,
                separate_folder=False,
                output_dtype=np.uint8,
                output_ext=".png",
            )
            if save
            else Identity(),
            # AsDiscreted(keys=("pred", "seg"), to_onehot=2),
        ]
