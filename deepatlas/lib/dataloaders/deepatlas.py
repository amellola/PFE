"""Module for loading the data for the segmentation task."""
import logging
import os
from typing import Dict, List, Optional, cast

import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.data.utils import partition_dataset
from monai.transforms import Compose

from deepatlas.lib.dataloaders.segmentation import Segmentation as SegDataLoader

log = logging.getLogger(__name__)


def path_to_id(path):
    """Convert a path to an id."""
    return os.path.basename(path).split(".")[0]


class Deepatlas:
    """Module for loading the data for the segmentation task."""

    def __init__(
        self, data_dir: str, dataloader_seg: SegDataLoader, conf_maps=False
    ) -> None:
        """Initialize the segmentation dataloader."""
        self.data_dir: str = data_dir
        self.dataloader_seg: SegDataLoader = dataloader_seg

        self.data_pairs_train_subdivided: Optional[List[Dict[str, str]]] = None
        self.data_pairs_valid_subdivided: Optional[List[Dict[str, str]]] = None

        self.data_pairs_train: Optional[List[Dict[str, str]]] = None
        self.data_pairs_valid: Optional[List[Dict[str, str]]] = None

        self.seg_availabilities = None
        self.batch_generator_train_reg = None
        self.batch_generator_valid_reg = None
        self.batch_generator_train_seg = None
        self.dataset_pairs_valid_subdivided = None
        self.dataset_pairs_train_subdivided = None

        self.prepare(conf_maps)

    def load(
        self,
        *,
        batch_size=2,
        test_batch_size=2,
        cache_num=32,
        transforms_deepatlas,
        load_seg=False,
        transforms_seg=None,
    ):
        """Load the data in cached Datasets."""

        if load_seg:
            self.dataloader_seg.load(
                cache_num=cache_num,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                seg_transforms=transforms_seg,
            )

        transform_pair = Compose(transforms=transforms_deepatlas)

        log.info("loading joint-train data:")
        self.dataset_pairs_train_subdivided = {
            seg_availability: CacheDataset(
                data=data_list,
                transform=transform_pair,
                cache_num=cache_num,
            )
            for seg_availability, data_list in self.data_pairs_train_subdivided.items()
        }

        log.info("loading joint-valid data:")
        self.dataset_pairs_valid_subdivided = {
            seg_availability: CacheDataset(
                data=data_list,
                transform=transform_pair,
                cache_num=cache_num,
            )
            for seg_availability, data_list in self.data_pairs_valid_subdivided.items()
        }

        # The following are dictionaries that map segmentation availability labels 00,10,01,11 to MONAI dataloaders

        dataloader_pairs_train_subdivided = {
            seg_availability: DataLoader(
                dataset,
                batch_size=batch_size,
                # num_workers=4,
                shuffle=True,
            )
            if len(dataset) > 0
            else []  # empty dataloaders are not a thing-- put an empty list if needed
            for seg_availability, dataset in self.dataset_pairs_train_subdivided.items()
        }

        dataloader_pairs_valid_subdivided = {
            seg_availability: DataLoader(
                dataset,
                batch_size=test_batch_size,
                # num_workers=4,
                shuffle=True,  # Shuffle validation data because we will only take a sample for validation each time
            )
            if len(dataset) > 0
            else []  # empty dataloaders are not a thing-- put an empty list if needed
            for seg_availability, dataset in self.dataset_pairs_valid_subdivided.items()
        }

        self.seg_availabilities = ["00", "01", "10", "11"]

        self.batch_generator_train_reg = self._create_batch_generator(
            dataloader_pairs_train_subdivided
        )
        self.batch_generator_valid_reg = self._create_batch_generator(
            dataloader_pairs_valid_subdivided
        )

        # When training seg_net alone, we only consider data pairs for which at least one ground truth seg is available
        seg_train_sampling_weights = [0] + [
            len(dataloader_pairs_train_subdivided[s])
            for s in self.seg_availabilities[1:]
        ]
        log.info(
            "When training seg_net alone, segmentation availabilities %s will be sampled with respective weights %s",
            self.seg_availabilities,
            seg_train_sampling_weights,
        )

        self.batch_generator_train_seg = self._create_batch_generator(
            dataloader_pairs_train_subdivided, seg_train_sampling_weights
        )

    def prepare(self, conf_maps=False) -> None:
        """Crawl folders to find images and extract the paths.

        args:
            limit: limit the number of images to load
        """

        self.dataloader_seg.data_seg_unavailable = cast(
            List[Dict[str, str]], self.dataloader_seg.data_seg_unavailable
        )
        self.dataloader_seg.data_seg_available_train = cast(
            List[Dict[str, str]], self.dataloader_seg.data_seg_available_train
        )

        # During the joint/alternating training process, we will use reuse data_seg_available_valid
        # for validating the segmentation network.
        # So we should not let the registration or segmentation networks see these images in training.
        data_without_seg_valid = (
            self.dataloader_seg.data_seg_unavailable
            + self.dataloader_seg.data_seg_available_train
        )  # Note the order

        # For validation of the registration network, we prefer not to use the precious data_seg_available_train,
        # if that's possible. The following split tries to use data_seg_unavailable for the
        # the validation set, to the extent possible.
        # pylint: disable-next=unbalanced-tuple-unpacking
        data_valid, data_train = partition_dataset(
            data_without_seg_valid,  # Note the order
            ratios=(2, 8),  # Note the order
            shuffle=False,
        )

        self.data_pairs_valid = self._take_data_pairs(data_valid, conf_maps=conf_maps)
        self.data_pairs_train = self._take_data_pairs(data_train, conf_maps=conf_maps)

        self.data_pairs_valid_subdivided = self._subdivide_list_of_data_pairs(
            self.data_pairs_valid
        )
        self.data_pairs_train_subdivided = self._subdivide_list_of_data_pairs(
            self.data_pairs_train
        )

        self.log_info()

    # ------------------ Helper functions ------------------

    def _create_batch_generator(self, dataloader_subdivided, weights=None):
        """
        Create a batch generator that samples data pairs with various segmentation availabilities.

        Arguments:
            dataloader_subdivided : a mapping from the labels in seg_availabilities to dataloaders
            weights : a list of probabilities, one for each label in seg_availabilities;
                    if not provided then we weight by the number of data items of each type,
                    effectively sampling uniformly over the union of the datasets

        Returns: batch_generator
            A function that accepts a number of batches to sample and that returns a generator.
            The generator will weighted-randomly pick one of the seg_availabilities and
            yield the next batch from the corresponding dataloader.
        """
        if weights is None:
            weights = np.array(
                [len(dataloader_subdivided[s]) for s in self.seg_availabilities]
            )
        weights = np.array(weights)
        weights = weights / weights.sum()
        dataloader_subdivided_as_iterators = {
            s: iter(d) for s, d in dataloader_subdivided.items()
        }

        def batch_generator(num_batches_to_sample):
            if np.isnan(weights).any():
                return
            for _ in range(num_batches_to_sample):
                seg_availability = np.random.choice(self.seg_availabilities, p=weights)
                try:
                    yield next(dataloader_subdivided_as_iterators[seg_availability])
                except StopIteration:  # If dataloader runs out, restart it
                    dataloader_subdivided_as_iterators[seg_availability] = iter(
                        dataloader_subdivided[seg_availability]
                    )
                    yield next(dataloader_subdivided_as_iterators[seg_availability])

        return batch_generator

    def _take_data_pairs(
        self, data, conf_maps=False, symmetric=True
    ) -> List[Dict[str, str]]:
        """Split data into pairs of images.

        Given a list of dicts that have keys for an image and maybe a segmentation,
        return a list of dicts corresponding to *pairs* of images and maybe segmentations.
        Pairs consisting of a repeated image are not included.
        If symmetric is set to True, then for each pair that is included, its reverse is also included
        """
        data_pairs = []
        # pylint: disable-next=consider-using-enumerate
        for i in range(len(data)):
            j_limit = len(data) if symmetric else i
            for j in range(j_limit):
                if j == i:
                    continue
                data_01 = data[i]
                data_02 = data[j]
                pair = {"img1": data_01["img"], "img2": data_02["img"]}
                if conf_maps:
                    pair["cm1"] = data_01["cm"]
                    pair["cm2"] = data_02["cm"]
                if "seg" in data_01.keys():
                    pair["seg1"] = data_01["seg"]
                if "seg" in data_02.keys():
                    pair["seg2"] = data_02["seg"]
                data_pairs.append(pair)
        return data_pairs

    def _subdivide_list_of_data_pairs(self, data_pairs_list):
        """Sort the data pairs according to the number of segmentations available."""
        out_dict = {"00": [], "01": [], "10": [], "11": []}
        for d in data_pairs_list:
            if "seg1" in d.keys() and "seg2" in d.keys():
                out_dict["11"].append(d)
            elif "seg1" in d.keys():
                out_dict["10"].append(d)
            elif "seg2" in d.keys():
                out_dict["01"].append(d)
            else:
                out_dict["00"].append(d)
        return out_dict

    def log_info(self):
        """Write information about the dataset to log."""
        num_train_reg_net = len(self.data_pairs_train)
        num_valid_reg_net = len(self.data_pairs_valid)
        num_train_both = (
            len(self.data_pairs_train_subdivided["01"])
            + len(self.data_pairs_train_subdivided["10"])
            + len(self.data_pairs_train_subdivided["11"])
        )

        log.info(
            "We have %s pairs to train reg_net and seg_net together, and an additional %s to train reg_net alone.",
            num_train_both,
            num_train_reg_net - num_train_both,
        )
        log.info("We have %s pairs for reg_net validation.", num_valid_reg_net)
