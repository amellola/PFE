"""Specifies the transforms that can be applied to the data."""
from .deepatlas import Deepatlas as DeepatlasTransforms
from .segmentation import Segmentation as SegmentationTransforms
from .segmentation import transform_single as default_transform_single_dataset_3d_us
from .segmentation2d import Segmentation2D as Segmentation2DTransforms
