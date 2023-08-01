"""Defines the segmentation configuration for the DeepAtlas project."""
import logging
import os
from typing import Optional

import monai
import torch

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from deepatlas.lib.dataloaders import DeepatlasDataLoader, SegmentationDataLoader
from deepatlas.lib.infers import DeepatlasInferer
from deepatlas.lib.trainers import DeepatlasTrainer
from deepatlas.lib.transforms import (
    DeepatlasTransforms,
    Segmentation2DTransforms,
    SegmentationTransforms,
)

# pylint: enable=import-error
# pylint: enable=no-name-in-module

log = logging.getLogger(__name__)


class Deepatlas:
    """Config for Deepatlas joint Networks."""

    def __init__(self):
        """Initialize instance variables."""
        self.batch_size = None
        self.test_batch_size = None
        self.resize = None
        self.device = None
        self.model_dir = None
        self.seg_network = None
        self.reg_network = None
        self.data_dir = None
        self.output_dir = None
        self.num_segmentation_classes = None
        self.lr_seg = None
        self.lr_reg = None
        self.oasis = None
        self.cm_channel = None
        self.cm_loss = None
        self.size = None
        self.loss = None
        self.seg_transformer = None

    def init(
        self,
        *,
        num_segmentation_classes: int,
        device: str = "cuda",
        batch_size: int = 8,
        test_batch_size: int = 16,
        resize: Optional[int] = None,
        size: int = 512,
        model_dir: str = "./checkpoints",
        additional_model_dir: str = "./checkpoints-additional",
        output_dir: str = "./data",
        data_dir: str = "./datasets/deepatlas",
        lr_seg: float = 1e-3,
        lr_reg: float = 5e-4,
        oasis: bool = False,
        cm_channel: bool = False,
        cm_loss: bool = False,
        loss: str = "dice",
        seg_transformer: str = "default",
    ):
        """Initialize the segmentation configuration."""
        # parameters
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.resize = resize
        self.device = device
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.num_segmentation_classes = num_segmentation_classes
        self.lr_seg = lr_seg
        self.lr_reg = lr_reg
        self.oasis = oasis
        self.cm_channel = cm_channel
        self.cm_loss = cm_loss
        self.loss = loss
        self.size = size
        self.seg_transformer = seg_transformer

        self.seg_network = monai.networks.nets.UNet(
            3,  # spatial dims
            1 if not self.cm_channel else 2,  # input channels
            num_segmentation_classes,  # output channels
            (8, 16, 16, 32, 32, 64, 64),  # channel sequence
            (1, 2, 1, 2, 1, 2),  # convolutional strides
            dropout=0.2,
            norm="batch",
        )
        self.reg_network = monai.networks.nets.UNet(
            3,  # spatial dims
            2
            if not self.cm_channel
            else 4,  # input channels (one for fixed image and one for moving image) potentially two others for conf maps
            3,  # output channels (to represent 3D displacement vector field)
            (16, 32, 32, 32, 32),  # channel sequence
            (1, 2, 2, 2),  # convolutional strides
            dropout=0.2,
            norm="batch",
        )

        # Model Files TODO
        seg_path = [
            os.path.join(self.model_dir, "final_seg_net.pth"),  # published
            os.path.join(self.model_dir, "final_pretrained_seg_net.pth"),  # pretrained
        ]
        reg_path = [
            os.path.join(self.model_dir, "final_reg_net.pth"),
            os.path.join(self.model_dir, "final_reg_net.pth"),
        ]
        if additional_model_dir:
            seg_path += [
                os.path.join(additional_model_dir, "final_seg_net.pth"),
                os.path.join(additional_model_dir, "final_pretrained_seg_net.pth"),
            ]
            reg_path += [
                os.path.join(additional_model_dir, "final_reg_net.pth"),
            ]

        for path in seg_path:
            if os.path.exists(path):
                self.seg_network.load_state_dict(torch.load(path, map_location=device))
                log.info("Loaded pretrained segmentation network from %s", path)
                break
        for path in reg_path:
            if os.path.exists(path):
                self.reg_network.load_state_dict(torch.load(path, map_location=device))
                log.info("Loaded pretrained registration network from %s", path)
                break

        log.info("Deepatlas-Config initialized.")

    def infer(self):
        """Infer the segmentation network."""
        # return object of inference task
        return DeepatlasInferer(
            path=self.model_dir,
            device=self.device,
            resize=self.resize,
            seg_network=self.seg_network,
            reg_network=self.reg_network,
            conf_maps=self.cm_channel,
            size=self.size,
        )

    def trainer(self):
        """Train the segmentation network."""
        # return object of training task
        return DeepatlasTrainer(
            model_dir=self.model_dir,
            output_dir=self.output_dir,
            seg_network=self.seg_network,
            reg_network=self.reg_network,
            resize=self.resize,
            device=self.device,
            num_segmentation_classes=self.num_segmentation_classes,
            learning_rate_reg=self.lr_reg,
            learning_rate_seg=self.lr_seg,
            cm_channel=self.cm_channel,
            cm_loss=self.cm_loss,
            size=self.size,
            loss=self.loss,
        )

    def dataloader(self, limit_imgs: Optional[int] = None , limit_label: Optional[int] = None):
        """Load the data for training."""
        # return object of dataloader
        seg_dataloader = SegmentationDataLoader(
            self.data_dir,
            limit_imgs=limit_imgs,
            limit_label=limit_label,
            oasis=self.oasis,
            conf_maps=(self.cm_channel or self.cm_loss),
        )
        joint_dataloader = DeepatlasDataLoader(
            self.data_dir, seg_dataloader, conf_maps=(self.cm_channel or self.cm_loss)
        )
        return seg_dataloader, joint_dataloader

    def transformer(self):
        """Return the transformer."""
        if self.seg_transformer == "2d":
            seg_transformer = Segmentation2DTransforms(
                resize=self.resize,
                device=self.device,
                cm_channel=self.cm_channel,
                cm_loss=self.cm_loss,
                size=self.size,
            )
        else:
            seg_transformer = SegmentationTransforms(
                resize=self.resize,
                device=self.device,
                cm_channel=self.cm_channel,
                cm_loss=self.cm_loss,
                size=self.size,
            )
        joint_transformer = DeepatlasTransforms(
            resize=self.resize,
            cm_channel=self.cm_channel,
            size=self.size,
        )
        return seg_transformer, joint_transformer
