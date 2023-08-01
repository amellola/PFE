"""Implementation of the segmentation trainer."""

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from polyaxon_client.tracking import Experiment

import wandb
from deepatlas.lib.losses import SegmentationLoss


@dataclass
class Statistics:
    """Statistics of the segmentation training."""

    training_losses: List[List[complex]]
    validation_losses: List[List[complex]]


class Segmentation:
    """Trainer for Segmentation Network."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir,
        network: torch.nn.Module,
        description="",
        resize=128,
        size=512,
        device="cuda",
        lr=0.0001,
        cm_loss=False,
        cm_channel=False,
        num_segmentation_classes=4,
    ):
        """Initialize the trainer."""
        self._network = network
        self.resize = resize
        self.model_dir = model_dir
        self.description = description
        self.device = device
        self.lr = lr
        self.stats: Optional[Statistics] = None
        self.cm_loss = cm_loss
        self.cm_channel = cm_channel
        self.size = size
        self.num_segmentation_classes = num_segmentation_classes

    def network(self):
        """Return the network for the trainer."""
        return self._network

    def optimizer(self):
        """Return the optimizer for the trainer."""
        return torch.optim.Adam(self._network.parameters(), lr=self.lr)

    def loss_function(self):
        """Return the loss function for the trainer."""
        return SegmentationLoss(
            num_segmentation_classes=self.num_segmentation_classes, device=self.device
        )

    def train(  # pylint: disable=too-many-statements,too-many-locals
        self,
        *,
        epochs,
        writer,
        experiment: Experiment,
        dataloader,
        pretrain=False,
        save_interval=5,
        loss="dice",
    ):
        """Return the training configuration for the trainer."""
        network = self.network()
        optimizer = self.optimizer()
        seg_losses = self.loss_function()
        if loss == "dice":
            loss_function = seg_losses.dice_loss
        elif loss == "cross_entropy":
            loss_function = seg_losses.ce_loss
        elif loss == "dice_ce":
            loss_function = seg_losses.dice_ce_loss
        elif loss == "focal_loss":
            loss_function = seg_losses.focal_loss

        network.to(self.device)

        max_epochs = epochs
        training_losses = []
        validation_losses = []
        val_interval = 5
        best = 1e9

        for epoch_number in range(max_epochs):

            print(f"Epoch {epoch_number+1}/{max_epochs}:")

            network.train()
            losses = []
            
            for batch in dataloader.dataloader_seg_available_train:

                # print("Loading images to GPU...")

                imgs = batch["img"].to(self.device) 
                # print("Training on:")
                # print(batch["img"].meta["filename_or_obj"][0])

                true_segs = batch["seg"].to(self.device)

                optimizer.zero_grad()
                predicted_segs = network(imgs)
                cm = None
                if self.cm_loss:
                    if self.cm_channel:
                        if len(imgs.shape) == 5:
                            cm = imgs[:, 1:2, :, :, :]
                        elif len(imgs.shape) == 4:
                            cm = imgs[:, 1:2, :, :]
                    else:
                        cm = batch["cm"].to(self.device)
                loss = loss_function(predicted_segs, true_segs, cm)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # if imgs and true_segs and predicted_segs:
                #     del imgs, true_segs, predicted_segs

            training_loss = np.mean(losses)
            print(f"\ttraining loss: {training_loss}")
            training_losses.append([epoch_number, training_loss])

            # Save checkpoint
            if epoch_number % save_interval == 0:
                self.save_checkpoint(
                    is_best=training_loss < best, epoch=epoch_number, pretrain=pretrain
                )
            if training_loss < best:
                best = training_loss

            if epoch_number % val_interval == 0:
                network.eval()
                losses = []
                with torch.no_grad():
                    for batch in dataloader.dataloader_seg_available_valid:
                        imgs = batch["img"].to(self.device)

                        # print("Validating on:")
                        # print(batch["img"].meta["filename_or_obj"][0])

                        true_segs = batch["seg"].to(self.device)
                        predicted_segs = network(imgs)
                        cm = None
                        if self.cm_loss:
                            if self.cm_channel:
                                if len(imgs.shape) == 5:
                                    cm = imgs[:, 1:2, :, :, :]
                                elif len(imgs.shape) == 4:
                                    cm = imgs[:, 1:2, :, :]
                            else:
                                cm = batch["cm"].to(self.device)
                        loss = loss_function(predicted_segs, true_segs, cm)
                        losses.append(loss.item())

                        # if imgs and true_segs and predicted_segs:
                        #     del imgs, true_segs, predicted_segs

                validation_loss = np.mean(losses)
                print(f"\tvalidation loss: {validation_loss}")
                
                if pretrain:
                    to_log = {
                        "loss/pre/train/seg": training_loss,
                        "loss/pre/valid/seg": validation_loss,
                        "loss/combined/train/seg": training_loss,
                        "loss/combined/valid/seg": validation_loss,
                    }
                else:
                    to_log = {
                        "loss/train/seg": training_loss,
                        "loss/valid/seg": validation_loss,
                        "loss/combined/train/seg": training_loss,
                        "loss/combined/valid/seg": validation_loss,
                    }

                for key, value in to_log.items():
                    writer.add_scalar(key, value, epoch_number)

                wandb.log(  # type: ignore
                    to_log,
                    step=epoch_number,
                )
                experiment.log_metrics(
                    **to_log,
                    step=epoch_number,
                )

            else:
                if pretrain:
                    to_log = {
                        "loss/pre/train/seg": training_loss,
                        "loss/combined/train/seg": training_loss,
                    }
                else:
                    to_log = {
                        "loss/train/seg": training_loss,
                        "loss/combined/train/seg": training_loss,
                    }

                for key, value in to_log.items():
                    writer.add_scalar(key, value, epoch_number)

                wandb.log(  # type: ignore
                    to_log,
                    step=epoch_number,
                )
                experiment.log_metrics(
                    **to_log,
                    step=epoch_number,
                )

        self.save_checkpoint(pretrain=pretrain)

        # Free up some memory
        del loss, predicted_segs, true_segs, imgs
        torch.cuda.empty_cache()

        self.stats = Statistics(training_losses, validation_losses)

        return True
    
    def load_checkpoint(self, pretrain=False, path=None):
        """Load pth file to the network."""
        if path is None:
            path = f"{self.model_dir}/{'pretrained_' if pretrain else ''}seg_net.pth"
        else:
            path = f"{path}/{'pretrained_' if pretrain else ''}seg_net.pth"
        self._network.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )
        print("== checkpoint loaded")

    def save_checkpoint(self, *, is_best=False, epoch=-1, pretrain=False):
        """Save network state to pth file."""
        filepath = os.path.join(
            self.model_dir,
            f"{epoch if epoch >= 0 else 'final'}_{'pretrained_' if pretrain else ''}seg_net.pth",
        )
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self._network.state_dict(), filepath)
        if is_best:
            shutil.copyfile(
                filepath,
                os.path.join(
                    self.model_dir,
                    f"best_{'pretrained_' if pretrain else ''}seg_net.pth",
                ),
            )

        print(" == checkpoint saved")

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
