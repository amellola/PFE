"""Specified deepatlas transforms."""
# pylint: disable=unused-import
from .additional import PrintD

# pylint: disable-next=wrong-import-order
from monai.transforms import (  # isort:skip
    EnsureChannelFirstD,
    Identity,
    LoadImageD,
    ResizeD,
    ToTensorD,
    TransposeD,
    ConcatItemsD,
    DeleteItemsD,
    CenterSpatialCropD,
    SpatialPadD,
)


class Deepatlas:
    """Holds all transforms for segmentation with DeepAtlas networks."""

    def __init__(self, resize, cm_channel, size):
        """Initialize the transformer."""
        self.resize = resize
        self.cm_channel = cm_channel
        self.size = size

    def default(self):
        """Specify deepatlas transforms."""
        return [
            LoadImageD(
                keys=["img1", "seg1", "img2", "seg2"],
                image_only=True,
                allow_missing_keys=True,
            ),
            ToTensorD(keys=["img1", "seg1", "img2", "seg2"], allow_missing_keys=True),
            TransposeD(
                keys=["img1", "seg1", "img2", "seg2"],
                indices=(2, 1, 0),
                allow_missing_keys=True,
            ),
            EnsureChannelFirstD(
                keys=["img1", "seg1", "img2", "seg2"], allow_missing_keys=True
            ),
            ConcatItemsD(keys=["img1", "img2"], name="img12", dim=0),
            DeleteItemsD(keys=["img1", "img2"]),
            ResizeD(
                keys=["img12", "seg1", "seg2"],
                spatial_size=(self.resize, self.resize, self.resize),
                mode=["trilinear", "nearest", "nearest"],
                allow_missing_keys=True,
                align_corners=[False, None, None],
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
            # PrintD(keys=["img1", "img2"]),
            LoadImageD(
                keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
                image_only=True,
                allow_missing_keys=True,
            ),
            ToTensorD(
                keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
                allow_missing_keys=True,
            ),
            TransposeD(
                keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
                indices=axis_order,
                allow_missing_keys=True,
            ),
            # CleanTransformD(keys=["seg1", "seg2"], only_sol=False, allow_missing_keys=True),
            EnsureChannelFirstD(
                keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
                allow_missing_keys=True,
            ),
            # CenterSpatialCropD(
            #     keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
            #     roi_size=(-1, h, w),
            #     allow_missing_keys=True,
            # ),
            # SpatialPadD(
            #     keys=["img1", "seg1", "img2", "seg2", "cm1", "cm2"],
            #     spatial_size=(-1, h, w),
            #     allow_missing_keys=True,
            # ),
            ResizeD(
                keys=["img1", "img2", "seg1", "seg2", "cm1", "cm2"],
                spatial_size=spatial_size,
                mode=[
                    "trilinear",
                    "trilinear",
                    "nearest",
                    "nearest",
                    "trilinear",
                    "trilinear",
                ],
                allow_missing_keys=True,
                align_corners=[False, False, None, None, False, False],
            )
            if self.resize is not None
            else Identity(),
            # === Confidence Maps ===
            ConcatItemsD(keys=("img1", "cm1"), name="img1", dim=0)
            if self.cm_channel
            else Identity(),
            ConcatItemsD(keys=("img2", "cm2"), name="img2", dim=0)
            if self.cm_channel
            else Identity(),
            DeleteItemsD(keys=("cm1", "cm2")) if self.cm_channel else Identity(),
            # =======================
            ConcatItemsD(keys=["img1", "img2"], name="img12", dim=0),
            DeleteItemsD(keys=["img1", "img2"]),
        ]
