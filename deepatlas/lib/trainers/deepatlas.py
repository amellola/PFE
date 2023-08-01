"""Implementation of the segmentation trainer."""

import logging
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import monai
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from polyaxon_client.tracking import Experiment

import wandb

# pylint: disable-next=import-error
# pylint: disable-next=no-name-in-module
from deepatlas.lib.losses import DeepatlasLoss

log = logging.getLogger(__name__)


@dataclass
class Statistics:
    """Statistics of the joint registration and segmentation training."""

    training_losses_reg: List[List[complex]]
    validation_losses_reg: List[List[complex]]
    training_losses_seg: List[List[complex]]
    validation_losses_seg: List[List[complex]]


@dataclass
class Context:
    """Holds context for trainging."""

    lambda_a: float
    lambda_sp: float
    lambda_r_multiplier: float

    reg_phase_training_batches_per_epoch: int
    seg_phase_training_batches_per_epoch: int
    reg_phase_num_validation_batches_to_use: int
    val_interval: int


class Deepatlas:
    """Trainer for Segmentation Network."""

    def __init__(
        self,
        *,
        model_dir,
        output_dir,
        reg_network: torch.nn.Module,
        seg_network: torch.nn.Module,
        num_segmentation_classes: int,
        description="",
        resize=128,
        size=512,
        device="cuda",
        learning_rate_reg=5e-4,
        learning_rate_seg=1e-3,
        cm_channel=False,
        cm_loss=False,
        loss="dice",
    ):
        """Initialize the trainer."""
        self._reg_network = reg_network
        self._seg_network = seg_network
        self.resize = resize
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.description = description
        self.device = device
        self.seg_lr = learning_rate_seg
        self.reg_lr = learning_rate_reg
        self.stats: Optional[Statistics] = None
        self.num_segmentation_classes = num_segmentation_classes
        self.cm_channel = cm_channel
        self.cm_loss = cm_loss
        self.size = size
        self.loss = loss

    def network(self):
        """Return the network for the trainer."""
        return self._seg_network, self._reg_network

    def optimizer(self):
        """Return the optimizer for the trainer."""
        return torch.optim.Adam(
            self._seg_network.parameters(), lr=self.seg_lr
        ), torch.optim.Adam(self._reg_network.parameters(), lr=self.reg_lr)

    def loss_function(self):
        """Return the loss function for the trainer."""
        return DeepatlasLoss(
            num_segmentation_classes=self.num_segmentation_classes, loss=self.loss
        )

    def take_random_from_subdivided_dataset(self, dataset_subdivided):
        """Given a dict mapping segmentation availability labels to datasets, return a random data item."""
        datasets = list(dataset_subdivided.values())
        datasets_combined = sum(datasets[1:], datasets[0])
        return random.choice(datasets_combined)

    def calc_img_scale(self, dataloader):
        """Calculate the image scale for the given dataloader."""
        data_item = self.take_random_from_subdivided_dataset(
            dataloader.dataset_pairs_train_subdivided
        )
        reg_net_example_input = data_item["img12"].unsqueeze(0)
        reg_net_example_output = self._reg_network(
            reg_net_example_input.to(self.device)
        )
        log.info("Shape of reg_net input: %s", reg_net_example_input.shape)
        log.info("Shape of reg_net output: %s", reg_net_example_output.shape)
        return reg_net_example_input.shape[-1]  # comes in handy later

    def _swap_training(self, network_to_train, network_to_not_train):
        """Switch out of training one network and into training another."""
        for param in network_to_not_train.parameters():
            param.requires_grad = False

        for param in network_to_train.parameters():
            param.requires_grad = True

        network_to_not_train.eval()
        network_to_train.train()

    def _get_cms(self, batch):
        """Get the confidence maps from batch."""
        cm1 = None
        cm2 = None
        if self.cm_loss:
            if self.cm_channel:
                cm1 = batch["img12"].to(self.device)[:, 1, :, :, :]
                cm2 = batch["img12"].to(self.device)[:, 3, :, :, :]
            else:
                cm1 = batch["cm1"].to(self.device)
                cm2 = batch["cm2"].to(self.device)
        return cm1, cm2

    def _get_cm1(self, batch):
        """Get the confidence maps from batch."""
        cm1 = None
        if self.cm_loss:
            if self.cm_channel:
                cm1 = batch["img12"].to(self.device)[:, 1, :, :, :]
            else:
                cm1 = batch["cm1"].to(self.device)
        return cm1

    def _get_cm2(self, batch):
        """Get the confidence maps from batch."""
        cm2 = None
        if self.cm_loss:
            if self.cm_channel:
                cm2 = batch["img12"].to(self.device)[:, 3, :, :, :]
            else:
                cm2 = batch["cm2"].to(self.device)
        return cm2

    def train(
        self,
        *,
        max_epochs,
        writer=None,
        experiment: Experiment = None,
        dataloader,
        context: Context = Context(
            lambda_a=2.0,  # anatomy loss weight
            lambda_sp=3.0,  # supervised segmentation loss weight
            lambda_r_multiplier=7.5,  # regularization loss weight - multiplier
            reg_phase_training_batches_per_epoch=40,
            seg_phase_training_batches_per_epoch=5,  # Fewer batches needed, because seg_net converges more quickly
            reg_phase_num_validation_batches_to_use=40,
            val_interval=5,
        ),
        pretrained_epochs: int = 0,
    ):
        """Train the network.

        Args:
            max_epochs: maximum number of epochs to train for
            dataloader: dataloader with loaded data
            writer: SummaryWriter for tensorboard. Defaults to None.
            context (Context, optional): Train context with relevant values. Defaults to
                Context(
                    lambda_a=2.0,  # anatomy loss weight
                    lambda_sp=3.0,  # supervised segmentation loss weight
                    lambda_r_multiplier=7.5,  # regularization loss weight - multiplier
                    reg_phase_training_batches_per_epoch=40,
                    seg_phase_training_batches_per_epoch=5,  # Fewer batches needed, because seg_net converges more quickly
                    reg_phase_num_validation_batches_to_use=40, val_interval=5,
                    ).
        """
        seg_net, reg_net = self.network()

        seg_net.to(self.device)
        reg_net.to(self.device)

        optimizer_seg, optimizer_reg = self.optimizer()
        deepatlas_losses = self.loss_function()

        image_scale = self.calc_img_scale(dataloader=dataloader)

        # regularization loss weight
        # This often requires some careful tuning. Here we suggest a value, which unfortunately needs to
        # depend on image scale. This is because the bending energy loss is not scale-invariant.
        # 7.5 worked well with the above hyperparameters for images of size 128x128x128.
        lambda_r = context.lambda_r_multiplier * (image_scale / self.resize) ** 2

        training_losses_reg = []
        validation_losses_reg = []
        training_losses_seg = []
        validation_losses_seg = []

        best_seg_validation_loss = float("inf")
        best_reg_validation_loss = float("inf")

        for epoch_number in range(max_epochs):

            log.info("Epoch %s/%s:", epoch_number + 1, max_epochs)

            # ------------------------------------------------
            #         reg_net training, with seg_net frozen
            # ------------------------------------------------

            # Keep computational graph in memory for reg_net, but not for seg_net, and do reg_net.train()
            self._swap_training(reg_net, seg_net)

            losses = []
            for batch in dataloader.batch_generator_train_reg(
                context.reg_phase_training_batches_per_epoch
            ):
                optimizer_reg.zero_grad()
                img12 = batch["img12"].to(self.device)
                loss_sim, loss_reg, loss_ana = deepatlas_losses.reg_losses(
                    batch, self.device, reg_net, seg_net, cm_channel=self.cm_channel
                )
                loss = loss_sim + lambda_r * loss_reg + context.lambda_a * loss_ana
                loss.backward()
                optimizer_reg.step()
                losses.append(loss.item())

            training_loss = np.mean(losses)
            log.info("\treg training loss: %s", training_loss)
            training_losses_reg.append([epoch_number, training_loss])

            writer.add_scalar("loss/joint/train/reg", training_loss, epoch_number)
            wandb.log({"loss/joint/train/reg": training_loss}, step=epoch_number)  # type: ignore
            if experiment:
                experiment.log_metrics(
                    **{"loss/joint/train/reg": training_loss}, step=epoch_number
                )

            if epoch_number % context.val_interval == 0:
                reg_net.eval()
                losses = []
                with torch.no_grad():
                    for batch in dataloader.batch_generator_valid_reg(
                        context.reg_phase_num_validation_batches_to_use
                    ):
                        loss_sim, loss_reg, loss_ana = deepatlas_losses.reg_losses(
                            batch,
                            self.device,
                            reg_net,
                            seg_net,
                            cm_channel=self.cm_channel,
                        )
                        loss = (
                            loss_sim + lambda_r * loss_reg + context.lambda_a * loss_ana
                        )
                        losses.append(loss.item())

                validation_loss = np.mean(losses)
                log.info("\treg validation loss: %s", validation_loss)
                validation_losses_reg.append([epoch_number, validation_loss])

                writer.add_scalar("loss/joint/valid/reg", validation_loss, epoch_number)
                wandb.log({"loss/valid/reg": validation_loss}, step=epoch_number)  # type: ignore
                if experiment:
                    experiment.log_metrics(
                        **{"loss/valid/reg": validation_loss}, step=epoch_number
                    )

                self.save_checkpoint(
                    network=seg_net,
                    epoch=epoch_number,
                    is_best=validation_loss < best_reg_validation_loss,
                    name="seg_net",
                )
                self.save_checkpoint(
                    network=reg_net,
                    epoch=epoch_number,
                    is_best=validation_loss < best_reg_validation_loss,
                    name="reg_net",
                )

                if validation_loss < best_reg_validation_loss:
                    best_reg_validation_loss = validation_loss

            # Free up memory
            del loss, loss_sim, loss_reg, loss_ana
            torch.cuda.empty_cache()

            # ------------------------------------------------
            #         seg_net training, with reg_net frozen
            # ------------------------------------------------

            # Keep computational graph in memory for seg_net, but not for reg_net, and do seg_net.train()
            self._swap_training(seg_net, reg_net)

            losses = []
            for batch in dataloader.batch_generator_train_seg(
                context.seg_phase_training_batches_per_epoch
            ):
                optimizer_seg.zero_grad()
                img12 = batch["img12"].to(self.device)

                displacement_fields = reg_net(img12)
                if self.cm_channel:
                    seg1_predicted = seg_net(img12[:, :2, :, :, :]).softmax(dim=1)
                    seg2_predicted = seg_net(img12[:, 2:, :, :, :]).softmax(dim=1)
                else:
                    seg1_predicted = seg_net(img12[:, [0], :, :, :]).softmax(dim=1)
                    seg2_predicted = seg_net(img12[:, [1], :, :, :]).softmax(dim=1)

                # Below we compute the following:
                # loss_supervised: supervised segmentation loss; compares ground truth seg with predicted seg
                # loss_anatomy: anatomy loss; compares warped seg of moving image to seg of target image
                # loss_metric: a single supervised seg loss, as a metric to track the progress of training

                if "seg1" in batch.keys() and "seg2" in batch.keys():
                    seg1 = monai.networks.one_hot(
                        batch["seg1"].to(self.device), self.num_segmentation_classes
                    )
                    seg2 = monai.networks.one_hot(
                        batch["seg2"].to(self.device), self.num_segmentation_classes
                    )
                    cm1, cm2 = self._get_cms(batch)
                    loss_metric = deepatlas_losses.seg_train_loss(
                        seg2_predicted, seg2, cm=cm2
                    )
                    loss_supervised = (
                        deepatlas_losses.seg_train_loss(seg1_predicted, seg1, cm=cm1)
                        + loss_metric
                    )
                    # The above supervised loss looks a bit different from the one in the paper
                    # in that it includes predictions for both images in the current image pair;
                    # we might as well do this, since we have gone to the trouble of loading
                    # both segmentations into memory.

                elif "seg1" in batch.keys():  # seg1 available, but no seg2
                    seg1 = monai.networks.one_hot(
                        batch["seg1"].to(self.device), self.num_segmentation_classes
                    )
                    cm1 = self._get_cm1(batch)
                    loss_metric = deepatlas_losses.seg_train_loss(
                        seg1_predicted, seg1, cm=cm1
                    )
                    loss_supervised = loss_metric
                    seg2 = seg2_predicted  # Use this in anatomy loss

                else:  # seg2 available, but no seg1
                    assert "seg2" in batch.keys()
                    seg2 = monai.networks.one_hot(
                        batch["seg2"].to(self.device), self.num_segmentation_classes
                    )
                    cm2 = self._get_cm2(batch)
                    loss_metric = deepatlas_losses.seg_train_loss(
                        seg2_predicted, seg2, cm=cm2
                    )
                    loss_supervised = loss_metric
                    seg1 = seg1_predicted  # Use this in anatomy loss

                # seg1 and seg2 should now be in the form of one-hot class probabilities
                cm1 = self._get_cm1(batch)
                loss_anatomy = (
                    deepatlas_losses.seg_train_loss(
                        deepatlas_losses.warp_nearest(seg2, displacement_fields),
                        seg1,
                        cm=cm1,
                    )
                    if "seg1" in batch.keys() or "seg2" in batch.keys()
                    else 0.0
                )  # It wouldn't really be 0, but it would not contribute to training seg_net

                # (If you want to refactor this code for *joint* training of reg_net and seg_net,
                #  then use the definition of anatomy loss given in the function anatomy_loss above,
                #  where differentiable warping is used and reg net can be trained with it.)

                loss = (
                    context.lambda_a * loss_anatomy
                    + context.lambda_sp * loss_supervised
                )
                loss.backward()
                optimizer_seg.step()

                losses.append(loss_metric.item())

            training_loss = np.mean(losses)
            log.info("\tseg training loss: %s", training_loss)
            training_losses_seg.append([epoch_number, training_loss])
            writer.add_scalar("loss/joint/train/seg", training_loss, epoch_number)
            writer.add_scalar(
                "loss/combined/train/seg",
                training_loss,
                pretrained_epochs + epoch_number,
            )
            wandb.log({"loss/joint/train/seg": training_loss}, step=epoch_number)  # type: ignore
            wandb.log(  # type: ignore
                {"loss/combined/train/seg": training_loss},
                step=pretrained_epochs + epoch_number,
            )
            experiment.log_metrics(
                **{"loss/joint/train/seg": training_loss}, step=epoch_number
            )
            experiment.log_metrics(
                **{"loss/combined/train/seg": training_loss},
                step=pretrained_epochs + epoch_number,
            )

            if epoch_number % context.val_interval == 0:
                # The following validation loop would not do anything in the case
                # where there is just one segmentation available,
                # because data_seg_available_valid would be empty.
                seg_net.eval()
                losses = []
                with torch.no_grad():
                    for (
                        batch
                    ) in dataloader.dataloader_seg.dataloader_seg_available_valid:
                        imgs = batch["img"].to(self.device)
                        true_segs = batch["seg"].to(self.device)
                        predicted_segs = seg_net(imgs)
                        loss = deepatlas_losses.dice_loss(predicted_segs, true_segs)
                        losses.append(loss.item())

                validation_loss = np.mean(losses)
                log.info("\tseg validation loss: %s", validation_loss)
                validation_losses_seg.append([epoch_number, validation_loss])

                writer.add_scalar("loss/joint/valid/seg", validation_loss, epoch_number)
                writer.add_scalar(
                    "loss/combined/valid/seg",
                    training_loss,
                    pretrained_epochs + epoch_number,
                )
                wandb.log({"loss/joint/valid/seg": validation_loss}, step=epoch_number)  # type: ignore
                wandb.log(  # type: ignore
                    {"loss/combined/valid/seg": validation_loss},
                    step=pretrained_epochs + epoch_number,
                )
                experiment.log_metrics(
                    **{"loss/joint/valid/seg": validation_loss}, step=epoch_number
                )
                experiment.log_metrics(
                    **{"loss/combined/valid/seg": validation_loss},
                    step=pretrained_epochs + epoch_number,
                )

                self.save_checkpoint(
                    network=seg_net,
                    epoch=epoch_number,
                    is_best=validation_loss < best_seg_validation_loss,
                    name="seg_net",
                )
                self.save_checkpoint(
                    network=reg_net,
                    epoch=epoch_number,
                    is_best=validation_loss < best_seg_validation_loss,
                    name="reg_net",
                )

                if validation_loss < best_seg_validation_loss:
                    best_seg_validation_loss = validation_loss

            # Free up memory
            del (
                loss,
                seg1,
                seg2,
                displacement_fields,
                img12,
                loss_supervised,
                loss_anatomy,
                loss_metric,
                seg1_predicted,
                seg2_predicted,
            )
            torch.cuda.empty_cache()

        log.info("Best reg_net validation loss: %s", best_reg_validation_loss)
        log.info("Best seg_net validation loss: %s", best_seg_validation_loss)

        self.save_checkpoint(network=seg_net, name="seg_net")
        self.save_checkpoint(network=reg_net, name="reg_net")

        self.stats = Statistics(
            training_losses_reg=training_losses_reg,
            validation_losses_reg=validation_losses_reg,
            training_losses_seg=training_losses_seg,
            validation_losses_seg=validation_losses_seg,
        )

    def load_checkpoint(self, *, network, name):
        """Load pth file to the network."""
        network.load_state_dict(
            torch.load(f"{self.model_dir}/{name}.pth", map_location=self.device)
        )
        log.info("== checkpoint loaded from: %s", self.model_dir)

    def save_checkpoint(self, *, network, is_best=False, epoch=-1, name):
        """Save network state to pth file."""
        filepath = os.path.join(
            self.model_dir, f"{epoch if epoch >= 0 else 'final'}_{name}.pth"
        )
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(network.state_dict(), filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(self.model_dir, f"best_{name}.pth"))

        log.info(" == checkpoint saved: %s", filepath)

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
