"""Module implementing the segmentation inference class."""
import logging

from typing import Callable, Sequence

import torch
from monai.inferers import Inferer, SlidingWindowInferer

log = logging.getLogger(__name__)

class Registration:
    """This provides Inference Engine for pre-trained segmentation model."""

    def __init__(
        self,
        path,
        network=None,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) registration",
        **kwargs,
    ):
        """Initialize the trainer."""
        self.path = path
        self.network = network
        self.labels = labels
        self.dimension = dimension
        self.description = description
        self.kwargs = kwargs

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        pass

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=(160, 160, 160))

    def post_transforms(self, data=None) -> Sequence[Callable]:
        pass

    def load_checkpoint(self, *, path):
        """Load pth file to the network."""
        self.network.load_state_dict(
            torch.load(path)
        )
        log.info("== checkpoint loaded from: %s", path)
