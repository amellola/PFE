"""Module implementing the segmentation inference class."""
import logging
from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer

from .registration import Registration as RegistrationInferer
from .segmentation import Segmentation as SegmentationInferer

log = logging.getLogger(__name__)


class Deepatlas:
    """This provides Inference Engine for pre-trained segmentation model."""

    def __init__(
        self,
        path,
        device="cuda",
        seg_network=None,
        reg_network=None,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation and registration",
        conf_maps=False,
        resize=128,
        size=512,
        **kwargs,
    ):
        """Initialize the trainer."""
        self.path = path
        self.seg_net = seg_network
        self.reg_net = reg_network
        self.labels = labels
        self.dimension = dimension
        self.description = description
        self.device = device
        self.kwargs = kwargs
        self.conf_maps = conf_maps
        self.resize = resize
        self.size = size

    def inferer(self, data=None):
        args = {
            "path": self.path,
            "labels": self.labels,
            "dimension": self.dimension,
            "device": self.device,
            "conf_maps": self.conf_maps,
            "load_cm": self.conf_maps,
            "resize": self.resize,
            "size": self.size,
        }
        seg_inferer = SegmentationInferer(**args, network=self.seg_net)
        reg_inferer = RegistrationInferer(**args, network=self.reg_net)
        return seg_inferer, reg_inferer
