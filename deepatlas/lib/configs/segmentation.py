"""Defines the segmentation configuration for the DeepAtlas project."""

import logging
import os
from typing import Optional

import monai
from monai.networks.nets import UNETR, AttentionUnet, UNet

from deepatlas.lib.dataloaders import SegmentationDataLoader
from deepatlas.lib.infers import SegmentationInferer
from deepatlas.lib.trainers import SegmentationTrainer
from deepatlas.lib.transforms import Segmentation2DTransforms, SegmentationTransforms

log = logging.getLogger(__name__)


class Segmentation:
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
        self.lr = None
        self.oasis = None
        self.data_dir = None
        self.cm_channel = None
        self.cm_loss = None
        self.size = None
        self.num_segmentation_classes = None
        self.transforms = None

    def init(
        self,
        *,
        device: str = "cuda",
        num_segmentation_classes: int,
        batch_size: int = 8,
        test_batch_size: int = 16,
        resize: Optional[int] = None,
        size: int = 512,
        model_dir: str = "./checkpoints",
        name: str = "segmentation",
        lr: float = 0.0001,
        network: str = "unet",
        oasis: bool = False,
        data_dir: str = "./datasets/dataset",
        cm_channel: bool = False,
        cm_loss: bool = False,
        transformer: str = "default",
    ):
        """Initialize the segmentation configuration."""
        # parameters
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.resize = resize
        self.device = device
        self.stats = None
        self.model_dir = model_dir
        self.lr = lr
        self.oasis = oasis
        self.data_dir = data_dir
        self.cm_channel = cm_channel
        self.cm_loss = cm_loss
        self.size = size
        self.num_segmentation_classes = num_segmentation_classes
        self.transforms = transformer
        self.network_str = network

        if network == "unet":
            self.network = UNet(
                3,  # spatial dims
                1 if not cm_channel else 2,  # input channels
                num_segmentation_classes,  # output channels
                (8, 16, 16, 32, 32, 64, 64),  # channel sequence
                (1, 2, 1, 2, 1, 2),  # convolutional strides
                dropout=0.2,
                norm="batch",
            )
        elif network == "unetr":
            self.network = UNETR(
                spatial_dims=3,
                in_channels=1 if not cm_channel else 2,  # input channels
                out_channels=num_segmentation_classes,  # output channels
                img_size=(resize, resize, resize),  # image size
                norm_name="batch",
            )
        elif network == "uneta":
            self.network = AttentionUnet(
                spatial_dims=3,  # spatial dims
                in_channels=1 if not cm_channel else 2,  # input channels
                out_channels=num_segmentation_classes,  # output channels
                channels=(8, 16, 32, 64),
                strides=(2, 2, 2),
                dropout=0.2,
            )
        elif network == "unet2d":
            log.info("Using 2D UNet")
            self.network = UNet(
                spatial_dims=2,  # spatial dims
                in_channels=1 if not cm_channel else 2,  # input channels
                out_channels=num_segmentation_classes,  # output channels
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                norm="batch",
            )

        # Model Files TODO
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pth"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pth"),  # published
        ]

    def infer(self):
        """Infer the segmentation network."""
        return SegmentationInferer(
            path=self.model_dir,
            device=self.device,
            resize=self.resize,
            network=self.network,
            dimension=2 if self.network_str == "unet2d" else 3,
            conf_maps=self.cm_channel,
            load_cm=(self.cm_channel or self.cm_loss),
            size=self.size,
        )

    def trainer(self):
        """Train the segmentation network."""
        return SegmentationTrainer(
            model_dir=self.model_dir,
            network=self.network,
            resize=self.resize,
            device=self.device,
            lr=self.lr,
            cm_loss=self.cm_loss,
            cm_channel=self.cm_channel,
            size=self.size,
            num_segmentation_classes=self.num_segmentation_classes,
        )

    def dataloader(self, limit_imgs: Optional[int] = None, limit_label: Optional[int] = None):
        """Load the data for training."""
        # return object of dataloader
        return SegmentationDataLoader(
            self.data_dir,
            limit_imgs=limit_imgs,
            limit_label=limit_label,
            oasis=self.oasis,
            conf_maps=(self.cm_channel or self.cm_loss),
        )

    def transformer(self):
        """Return the transformer."""
        if self.transforms == "2d":
            return Segmentation2DTransforms(
                resize=self.resize,
                device=self.device,
                cm_channel=self.cm_channel,
                cm_loss=self.cm_loss,
                size=self.size,
            )
        else:
            return SegmentationTransforms(
                resize=self.resize,
                device=self.device,
                cm_channel=self.cm_channel,
                cm_loss=self.cm_loss,
                size=self.size,
            )
