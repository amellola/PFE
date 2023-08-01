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
    RandFlipD,
    CropForegroundD,
    RandCropByPosNegLabelD,
    RandShiftIntensityD,
    SpacingD,
    RandRotate90D,
)

log = logging.getLogger(__name__)

import torch
from monai.transforms import Transform


class CleanTransform(Transform):
    def __init__(self, only_sol=False, skip=False):
        self.only_sol = only_sol
        self.skip = skip

    def __call__(self, data):
        if self.skip:
            return data

        if isinstance(data, dict):
            if "seg" in data:
                seg = data["seg"]
                torch.where(seg == 100, torch.tensor(1, dtype=seg.dtype), seg, out=seg)
                if self.only_sol:
                    torch.where(seg > 0, torch.tensor(1, dtype=seg.dtype), seg, out=seg)
                data["seg"] = seg
            return data
        else:
            raise ValueError("Data must be a dictionary.")



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
        spatial_size = (self.resize, self.resize, self.resize)
        axis_order = (2, 1, 0)
        return [
            LoadImageD(
                keys=("img", "seg", "cm"), image_only=True, allow_missing_keys=True
            ),
            ToTensorD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            TransposeD(
                keys=("img", "seg", "cm"), indices=axis_order, allow_missing_keys=True
            ),
            EnsureChannelFirstD(keys=("img", "seg", "cm"), allow_missing_keys=True),
            CleanTransform(only_sol=False),
            RandFlipD(keys=("img", "seg"), prob=0.5, spatial_axis=(0, 1, 2)),
            CropForegroundD(keys=("img", "seg"), source_key="seg"),
            RandCropByPosNegLabelD(
                keys=("img", "seg"),
                label_key="seg",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="img",
                image_threshold=0,
            ),
            RandShiftIntensityD(keys=("img",), offsets=0.1, prob=0.5),
            SpacingD(
                keys=("img", "seg"),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandRotate90D(keys=("img", "seg"), prob=0.5, max_k=3),
            ResizeD(
                keys=("img", "seg", "cm"),
                spatial_size=spatial_size,
                mode=["trilinear", "nearest", "trilinear"],
                align_corners=[False, None, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            NormalizeIntensityD(keys=("img",)),
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
        ]

    def pre_transforms(self) -> Sequence[Callable]:
        """Pre inference transforms."""
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
            ResizeD(
                keys=("img", "cm"),
                spatial_size=spatial_size,
                mode=["trilinear", "trilinear"],
                align_corners=[False, False],
                allow_missing_keys=True,
            )
            if self.resize is not None
            else Identity(),
            ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
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
        """Specify inverse transforms for post-processing."""
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
            AsDiscreted(keys=("pred", "seg"), to_onehot=4),
        ]


def transform_single(_resize, _mode, _align_corners, skip_label_transform=False):
    """Specify default transforms for single image."""
    h, w = 512, 512
    axis_order = (2, 1, 0)
    spatial_size = (self.resize, self.resize, self.resize)
    return [
        LoadImage(image_only=True),
        ToTensor(),
        Transpose(indices=axis_order),
        EnsureChannelFirst(),
        CleanTransform(only_sol=False, skip=skip_label_transform),
        CenterSpatialCrop((-1, h, w)),
        SpatialPad((-1, h, w)),
        RandFlipd(keys=("img", "seg", "cm"), prob=0.5),
        CropForegroundD(keys=("img", "seg", "cm"), source_key="img"),
        RandCropByPosNegLabeld(
            keys=("img", "seg"),
            label_key="seg",
            spatial_size=(self.resize, self.resize, self.resize),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="img",
            image_threshold=0,
        ),
        RandShiftIntensityD(keys="img", offsets=0.1, prob=0.5),
        SpacingD(keys=("img", "seg", "cm"), pixdim=[1.0, 1.0, 1.0], mode=("bilinear", "nearest", "bilinear")),
        RandRotate90D(keys=("img", "seg", "cm"), prob=0.5, max_k=3),
        ResizeD(
            keys=("img", "seg", "cm"),
            spatial_size=spatial_size,
            mode=["trilinear", "nearest", "trilinear"],
            align_corners=[False, None, False],
            allow_missing_keys=True,
        )
        if self.resize is not None
        else Identity(),
        NormalizeIntensityD(keys=("img",)),
        ConcatItemsD(keys=("img", "cm"), name="img", dim=0)
        if self.cm_channel
        else Identity(),
        DeleteItemsD(keys=("cm",)) if self.cm_channel else Identity(),
    ]
