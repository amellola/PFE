"""Defines the segmentation configuration for the DeepAtlas project."""


import os

import monai


class Segmenation:
    """Config for Segmentation Network."""

    def __init__(self):
        """Initialize instance variables."""
        self.batch_size = None
        self.test_batch_size = None
        self.resize = None
        self.device = None
        self.stats = None
        self.model_dir = None
        self.network = None
        self.path = None
        self.warp = None
        self.warp_nearest = None

    def init(
        self,
        *,
        device: str,
        batch_size: int,
        test_batch_size: int,
        resize: int,
        model_dir: str,
        name: str,
    ):
        """Initialize the segmentation configuration."""
        # parameters
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.resize = resize
        self.device = device
        self.stats = None
        self.model_dir = model_dir

        self.network = monai.networks.nets.UNet(
            3,  # spatial dims
            2,  # input channels (one for fixed image and one for moving image)
            3,  # output channels (to represent 3D displacement vector field)
            (16, 32, 32, 32, 32),  # channel sequence
            (1, 2, 2, 2),  # convolutional strides
            dropout=0.2,
            norm="batch",
        )

        # Model Files TODO
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        self.warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
        self.warp_nearest = monai.networks.blocks.Warp(
            mode="nearest", padding_mode="border"
        )

        # Others
        # TODO: add config with:
        # - strtobool, int, ...
        # - conf.get("value", "default")
        # - logger.info(...)

    def infer(self):
        """Infer the segmentation network."""
        # return object of inference task
        pass

    def trainer(self):
        """Train the segmentation network."""
        # return object of training task
        pass
