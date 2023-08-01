"""Module for loading the data for the segmentation task."""
import logging
import multiprocessing
import os
import random
import shutil
import tempfile
from glob import glob
from typing import Dict, List, Optional, cast

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.apps.utils import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.data.utils import partition_dataset
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
)
from tqdm import tqdm

log = logging.getLogger(__name__)


def path_to_id(path):
    """Convert a path to an id."""
    return os.path.basename(path).split(".")[0]


def oasis_path_to_id(path):
    return os.path.basename(path).strip("OAS1_")[:8]


class Segmentation:
    """Module for loading the data for the segmentation task."""

    def __init__(
        self,
        data_dir: str,
        limit_imgs: Optional[int] = None,
        limit_label: Optional[int] = None,
        oasis=False,
        conf_maps=False,
    ) -> None:
        """Initialize the segmentation dataloader."""
        self.data_dir: str = data_dir

        self.data_seg_available_train: Optional[List[Dict[str, str]]] = None
        self.data_seg_available_valid: Optional[List[Dict[str, str]]] = None
        self.data_seg_unavailable: Optional[List[Dict[str, str]]] = None
        self.dataset_seg_available_valid = None
        self.dataset_seg_available_train = None

        self.dataloader_seg_available_train = None
        self.dataloader_seg_available_valid = None

        if oasis:
            self.oasis_prepare_data()
        self.prepare(
            limit_imgs=limit_imgs, limit_label=limit_label, conf_maps=conf_maps
        )

    def load(
        self,
        *,
        batch_size=2,
        test_batch_size=2,
        cache_num=32,
        seg_transforms,
    ):
        """Load the data for the segmentation task to cache."""
        log.info("start loading seg-Dataset...")

        transform_seg_available = Compose(seg_transforms)

        log.info("loading seg-train data:")
        self.dataset_seg_available_train = CacheDataset(
            data=self.data_seg_available_train,
            transform=transform_seg_available,
            cache_num=cache_num,
        )

        log.info("loading seg-valid data:")
        self.dataset_seg_available_valid = CacheDataset(
            data=self.data_seg_available_valid,
            transform=transform_seg_available,
            cache_num=cache_num,
        )

        # cores = multiprocessing.cpu_count()
        # cores = min(multiprocessing.cpu_count(), batch_size)
        cores = 1
        log.info("Using %s cores", cores)

        self.dataloader_seg_available_train = DataLoader(
            self.dataset_seg_available_train,
            # batch_size=8,
            batch_size=batch_size,
            num_workers=cores,
            shuffle=True,
        )
        self.dataloader_seg_available_valid = DataLoader(
            self.dataset_seg_available_valid,
            # batch_size=16,
            batch_size=test_batch_size,
            num_workers=cores,
            shuffle=False,
        )
        log.info("done loading seg-Dataset.")

    def oasis_prepare_data(self) -> None:
        root_dir = tempfile.mkdtemp() if self.data_dir is None else self.data_dir
        data_dir = os.path.join(root_dir, "OASIS-1")
        log.info("Root directory: %s", root_dir)
        log.info("Data directory: %s", data_dir)

        # Download Dataset if not there yet
        resource = (
            "https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc1.tar.gz"
        )
        md5 = "c83e216ef8654a7cc9e2a30a4cdbe0cc"

        compressed_file = os.path.join(root_dir, "oasis_cross-sectional_disc1.tar.gz")
        if not os.path.exists(data_dir):
            download_and_extract(resource, compressed_file, data_dir, md5)

        image_path_expression = (
            "PROCESSED/MPRAGE/T88_111/OAS1_*_MR*_mpr_n*_anon_111_t88_masked_gfc.img"
        )
        segmentation_path_expression = (
            "FSL_SEG/OAS1_*_MR*_mpr_n*_anon_111_t88_masked_gfc_fseg.img"
        )

        # Expect either of two reasonable ways of organizing extracted data:
        # 1) <data_dir>/disc1/OAS1_0031_MR1/...
        # 2) <data_dir>/OAS1_0031_MR1/...
        image_paths = glob(os.path.join(data_dir, "*", image_path_expression))
        image_paths += glob(os.path.join(data_dir, "*/*", image_path_expression))
        segmentation_paths = glob(
            os.path.join(data_dir, "*", segmentation_path_expression)
        )
        segmentation_paths += glob(
            os.path.join(data_dir, "*/*", segmentation_path_expression)
        )

        for image_path in tqdm(image_paths):
            nii_path = os.path.join(
                self.data_dir, oasis_path_to_id(image_path) + ".nii"
            )

            img = sitk.ReadImage(image_path)
            sitk.WriteImage(img, nii_path)

        os.makedirs(self.data_dir + "/labels/final/", exist_ok=True)

        for segmentation_path in tqdm(segmentation_paths):
            nii_path = os.path.join(
                self.data_dir + "/labels/final/",
                oasis_path_to_id(segmentation_path) + ".nii",
            )

            img = sitk.ReadImage(segmentation_path)
            sitk.WriteImage(img, nii_path)

        img_ids = list(map(oasis_path_to_id, image_paths))
        seg_ids = list(map(oasis_path_to_id, segmentation_paths))
        data = []
        for img_index, img_id in enumerate(img_ids):
            data_item = {"img": image_paths[img_index]}
            if img_id in seg_ids:
                data_item["seg"] = segmentation_paths[seg_ids.index(img_id)]
            data.append(data_item)
        data_seg_available = list(filter(lambda d: "seg" in d.keys(), data))
        # pylint: disable-next=unbalanced-tuple-unpacking
        (
            train_dataset,
            test_dataset,
        ) = partition_dataset(data_seg_available, ratios=(9, 1))

        test_img_path = os.path.join(self.data_dir, "imagesTs")
        test_seg_path = os.path.join(self.data_dir, "imagesTs", "labels", "final")

        os.makedirs(test_img_path, exist_ok=True)
        os.makedirs(test_seg_path, exist_ok=True)

        # move test images and labels to test folder
        for item in test_dataset:
            img_path = os.path.join(
                self.data_dir, oasis_path_to_id(item["img"]) + ".nii"
            )
            seg_path = os.path.join(
                self.data_dir,
                "labels",
                "final",
                oasis_path_to_id(item["seg"]) + ".nii",
            )

            # Check if files exist from previous execution
            if not os.path.exists(
                os.path.join(test_img_path, oasis_path_to_id(item["img"]) + ".nii")
            ):
                shutil.move(img_path, test_img_path)
            else:
                os.remove(img_path)
            if not os.path.exists(
                os.path.join(test_seg_path, oasis_path_to_id(item["seg"]) + ".nii")
            ):
                shutil.move(seg_path, test_seg_path)
            else:
                os.remove(seg_path)

        # Remove old data if validation cut was different in last execution
        for item in train_dataset:
            img_path = os.path.join(
                test_img_path, oasis_path_to_id(item["img"]) + ".nii"
            )
            seg_path = os.path.join(
                test_seg_path, oasis_path_to_id(item["seg"]) + ".nii"
            )
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(seg_path):
                os.remove(seg_path)

    def prepare(
        self,
        limit_imgs: Optional[int] = None,
        limit_label: Optional[int] = None,
        conf_maps=False,
    ) -> None:
        """Prepare the available data for the segmentation task.

        args:
            limit: limit the number of images to load
        """
        log.info("Preparing dataset in %s", self.data_dir)

        mask_dirs = [
            os.path.join(self.data_dir, "labels/final"),
            os.path.join(self.data_dir, "labels/original"),
        ]
        train_images = []
        train_labels = []
        train_conf_maps = []
        extensions = [".nii.gz", ".nii", ".mha", ".img", ".tif", ".png"]

        for files in extensions:
            train_images.extend(glob(os.path.join(self.data_dir, "*" + files)))
            for mask_dir in mask_dirs:
                train_labels.extend(glob(os.path.join(mask_dir, "*" + files)))
            if conf_maps:
                train_conf_maps.extend(
                    glob(os.path.join(self.data_dir, "conf_maps", "*" + files))
                )

        log.info("Found %s images", len(train_images))
        log.info("Found %s masks", len(train_labels))
        log.info("Found %s conf_maps", len(train_conf_maps))

        np.random.shuffle(train_images)

        if limit_imgs is not None:
            train_images = train_images[:limit_imgs]

        np.random.shuffle(train_labels)
        train_labels = train_labels[:limit_label]

        log.info("Using %s images", len(train_images))
        log.info("Using %s masks", len(train_labels))

        seg_ids = list(map(path_to_id, train_labels))
        cm_ids = list(map(path_to_id, train_conf_maps))
        img_ids = map(path_to_id, train_images)
        data = []
        for img_index, img_id in enumerate(img_ids):
            data_item = {"img": train_images[img_index]}
            if img_id in seg_ids:
                data_item["seg"] = train_labels[seg_ids.index(img_id)]
            if conf_maps and img_id in cm_ids:
                data_item["cm"] = train_conf_maps[cm_ids.index(img_id)]
            data.append(data_item)

        # Loading Segmentation-Only Data as well for pretraining so we can reuse the validation data
        data_seg_available = list(filter(lambda d: "seg" in d.keys(), data))
        self.data_seg_unavailable = list(filter(lambda d: "seg" not in d.keys(), data))

        # pylint: disable-next=unbalanced-tuple-unpacking
        (
            self.data_seg_available_train,
            self.data_seg_available_valid,
        ) = partition_dataset(data_seg_available, ratios=(9, 1))

        log.info("Using %s training images", str(len(self.data_seg_available_train)))  # type: ignore
        log.info("Using %s validation images", str(len(self.data_seg_available_valid)))  # type: ignore

        self.data_seg_available_train = cast(
            List[Dict[str, str]], self.data_seg_available_train
        )

    def load_dataset_simple(
        self,
        pre_transforms,
        with_labels=True,
        conf_maps=False,
        label_path="/labels/final/",
        batch_size=1,
    ):
        """Load the dataset for the segmentation task."""
        images = []
        labels = []
        cmaps = []
        extensions = [".nii.gz", ".nii", ".mha", ".img", ".tif", ".png"]

        log.info("Loading dataset from %s", self.data_dir)
        log.info("Looking for labels in %s", self.data_dir + label_path)
        if conf_maps:
            log.info("Looking for conf_maps in %s", self.data_dir + "/conf_maps/")
        else:
            log.info("Not looking for conf_maps")

        for files in extensions:
            images.extend(glob(os.path.join(self.data_dir, "*" + files)))
            if with_labels:
                labels.extend(
                    glob(os.path.join(self.data_dir + "/" + label_path, "*" + files))
                )
            if conf_maps:
                cmaps.extend(
                    glob(os.path.join(self.data_dir, "conf_maps", "*" + files))
                )

        seg_ids = list(map(path_to_id, labels))
        img_ids = map(path_to_id, images)
        cm_ids = list(map(path_to_id, cmaps))
        data = []
        for img_index, img_id in enumerate(img_ids):
            data_item = {}
            if with_labels:
                if img_id in seg_ids:
                    data_item["seg"] = labels[seg_ids.index(img_id)]
                else:
                    continue
            if conf_maps and img_id in cm_ids:
                data_item["cm"] = cmaps[cm_ids.index(img_id)]
            data_item["img"] = images[img_index]
            data.append(data_item)

        log.info("Found %s usable images", len(data))

        pre_transforms = Compose(pre_transforms)

        test_org_ds = CacheDataset(data=data, transform=pre_transforms, cache_num=1)

        cores = 1
        log.info("Using %s cores", cores)

        return DataLoader(test_org_ds, batch_size=batch_size, num_workers=cores)

    def load_gt_and_pred(self, gt_dir, pred_dir):
        """Load the ground truth and prediction data for the segmentation task."""
        gt_images = []
        pred_images = []
        extensions = [".nii.gz", ".nii", ".mha", ".img", ".tif", ".png"]

        log.info("Loading dataset from %s", self.data_dir)

        for files in extensions:
            gt_images.extend(glob(os.path.join(gt_dir, "*" + files)))
            pred_images.extend(glob(os.path.join(pred_dir, "*" + files)))

        gt_ids = list(map(path_to_id, gt_images))
        pred_ids = map(path_to_id, pred_images)
        data = []
        for pred_index, pred_id in enumerate(pred_ids):
            data_item = {}
            if pred_id in gt_ids:
                data_item["seg"] = gt_images[gt_ids.index(pred_id)]
            else:
                continue
            data_item["pred"] = pred_images[pred_index]
            data.append(data_item)

        log.info("Found %s usable images", len(data))

        transforms = Compose(
            [
                LoadImaged(keys=["pred", "seg"], image_only=True),
                EnsureTyped(keys=["pred", "seg"]),
                AsDiscreted(keys=["pred", "seg"], to_onehot=7),
            ]
        )

        ds = Dataset(data=data, transform=transforms)

        cores = 1
        log.info("Using %s cores", cores)

        return DataLoader(ds, batch_size=1, num_workers=cores)

    def load_transform_save(
        self,
        data,
        transforms,
        output_dir,
        resize,
        with_labels=True,
        label_path="/labels/final/",
    ):
        """Load, transform and save the data."""
        images = []
        labels = []
        extensions = [".nii.gz", ".nii", ".mha", ".img", ".tif", ".png"]

        log.info("Loading dataset from %s", self.data_dir)
        log.info("Looking for labels in %s", self.data_dir + label_path)

        for files in extensions:
            images.extend(glob(os.path.join(self.data_dir, "*" + files)))
            if with_labels:
                labels.extend(
                    glob(os.path.join(self.data_dir + "/" + label_path, "*" + files))
                )

        img_transforms = transforms(
            resize, "trilinear", False, skip_label_transform=True
        )
        label_transforms = transforms(resize, "nearest", None)
        img_transforms = Compose(img_transforms)
        label_transforms = Compose(label_transforms)

        # cores = multiprocessing.cpu_count()
        cores = 1
        log.info("Using %s cores", cores)

        img_ds = Dataset(data=images, transform=img_transforms)
        img_dl = DataLoader(img_ds, batch_size=1, num_workers=cores)

        label_ds = Dataset(data=labels, transform=label_transforms)
        label_dl = DataLoader(label_ds, batch_size=1, num_workers=cores)

        for img in img_dl:
            path = os.path.join(
                output_dir,
                img.meta["filename_or_obj"][0]
                .split("/")[-1]
                .replace(".mha", ".nii.gz"),
            )
            img = np.squeeze(img)
            converted_array = np.array(img, dtype=np.float32)
            affine = np.eye(7)
            nifti_file = nib.Nifti1Image(converted_array, affine)
            nib.save(nifti_file, path)

        for img in label_dl:
            path = os.path.join(
                output_dir,
                "labels",
                "final",
                img.meta["filename_or_obj"][0]
                .split("/")[-1]
                .replace(".mha", ".nii.gz"),
            )
            img = torch.argmax(img, dim=1).detach().cpu()[:, :, :, :]
            log.info("Saving image with shape %s", img.shape)
            img = np.squeeze(img)
            log.info("Changed to shape %s", img.shape)
            converted_array = np.array(img, dtype=np.uint8)
            affine = np.eye(7)
            nifti_file = nib.Nifti1Image(converted_array, affine)
            nib.save(nifti_file, path)
