"""Implementation of the segmentation trainer."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from deepatlas.lib.transforms import default_transform

# pylint: enable=import-error
# pylint: enable=no-name-in-module


@dataclass
class Statistics:
    """Statistics of the segmentation training."""

    training_losses: List[List[complex]]
    validation_losses: List[List[complex]]


class Segmenation:
    """Trainer for Segmentation Network."""

    def __init__(
        self,
        model_dir,
        network: torch.nn.Module,
        description="",
        resize=24,
        device="cuda",
        lr=0.0001,
    ):
        """Initialize the trainer."""
        self._network = network
        self.resize = resize
        self.model_dir = model_dir
        self.description = description
        self.device = device
        self.lr = lr
        self.stats = None

    def network(self):
        """Return the network for the trainer."""
        return self._network

    def optimizer(self):
        """Return the optimizer for the trainer."""
        return torch.optim.Adam(self._network.parameters(), lr=self.lr)

    def loss_function(self):
        """Return the loss function for the trainer."""
        # TODO:
        return DiceLoss(
            include_background=True,
            to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
            softmax=True,  # Note that our segmentation network is missing the softmax at the end
            reduction="mean",
        )

    def transforms(self, _context: Dict[str, Any]):
        """Return the transforms for the trainer."""
        return default_transform(resize=self.resize)

    def train(self, *, epochs, writer, data_loader, val_data_loader):
        """Return the training configuration for the trainer."""
        network = self.network()
        optimizer = self.optimizer()
        loss_function = self.loss_function()

        network.to(self.device)

        max_epochs = epochs
        training_losses = []
        validation_losses = []
        val_interval = 5

        # TODO: ....

    def train_pre_transforms(self, _context: Dict[str, Any]):
        """Return the pre-transforms for the trainer."""
        raise NotImplementedError()

    def train_post_transforms(self, _context: Dict[str, Any]):
        """Return the post-transforms for the trainer."""
        raise NotImplementedError()

    def val_pre_transforms(self):
        """Return the pre-transforms for the trainer."""
        raise NotImplementedError()

    def val_inferer(self):
        """Return the inferer for the trainer."""
        return SlidingWindowInferer(
            roi_size=(160, 160, 160), sw_batch_size=1, overlap=0.25
        )
