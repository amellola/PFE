"""Module implementing the segmentation inference class."""
import logging

import torch
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.inferers import Inferer, SlidingWindowInferer

# pylint: disable=unused-import
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropD,
    Compose,
    ConcatItemsD,
    DataStatsd,
    DeleteItemsD,
    EnsureChannelFirstD,
    EnsureTyped,
    Identity,
    Invertd,
    LoadImageD,
    ResizeD,
    SaveImageD,
    SpatialPadD,
    ToTensorD,
    TransposeD,
)
from tqdm import tqdm

# from deepatlas.lib.transforms.default import SaveNibD

log = logging.getLogger(__name__)


class Segmentation:
    """This provides Inference Engine for pre-trained segmentation model."""

    def __init__(
        self,
        path,
        device="cuda",
        network=None,
        resize=128,
        size=512,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation",
        conf_maps=False,
        load_cm=False,
        **kwargs,
    ):
        """Initialize the trainer."""
        self.path = path
        self.network = network
        self.resize = resize
        self.labels = labels
        self.dimension = dimension
        self.description = description
        self.kwargs = kwargs
        self.device = device
        self.conf_maps = conf_maps
        self.load_cm = load_cm
        self.size = size

    def inferer(self) -> Inferer:
        """Init and get the inferer."""
        print(self.dimension)
        if self.dimension == 2:
            log.info("Using 2D Sliding Window Inferer")
            return SlidingWindowInferer(roi_size=(self.resize, self.resize))
        return SlidingWindowInferer(roi_size=(self.resize, self.resize, self.resize))

    def load_checkpoint(self, *, path):
        """Load pth file to the network."""
        log.info("Loading checkpoint from %s", path)
        log.info("Using device %s", self.device)
        self.network.load_state_dict(
            torch.load(path, map_location=torch.device(self.device))
        )
        log.info("== checkpoint loaded from: %s", path)

    def infer(
        self,
        infer,
        dataloader,
        transformer,
        output_dir,
        writer,
        metrics=None,
        experiment=None,
        save_ram=False,
        csv_dir=None,
        batch_size=1,
    ):
        """Infer the data."""
        pre_transforms = transformer.pre_transforms()

        inv_transforms = transformer.transforms_to_inverse()

        if inv_transforms is not None:
            pre_names = dict()
            for t in pre_transforms:
                pre_names[t.__class__.__name__] = t

            if len(inv_transforms) > 0:
                transforms_to_inverse = [
                    pre_names[n if isinstance(n, str) else n.__name__]
                    for n in inv_transforms
                ]
            else:
                transforms_to_inverse = pre_transforms

            transforms_to_inverse = Compose(transforms_to_inverse)

            inverse_transforms = Compose(
                transformer.inverse_transforms(transforms_to_inverse)
            )

        dataset = dataloader.load_dataset_simple(
            pre_transforms,
            label_path="/labels/final/",
            conf_maps=self.load_cm,
            batch_size=batch_size,
        )

        self.network.to(self.device)
        self.network.eval()

        # pre_transforms = Compose(pre_transforms)
        post_transforms = Compose(transformer.post_transforms(output_dir))

        with torch.no_grad():
            for data in tqdm(dataset):
                test_inputs = data["img"].to(self.device)
                data["pred"] = infer(test_inputs, self.network)

                if inv_transforms is not None:
                    data = [
                        post_transforms(inverse_transforms(i))
                        for i in decollate_batch(data)
                    ]
                else:
                    data = [post_transforms(i) for i in decollate_batch(data)]

                for d in data:
                    del d["img"]

                if metrics:
                    log.info("Calculating metrics...")
                    metrics.calc_metrics(data, writer, experiment, reduction="none")

                # Save RAM
                for d in data:
                    del d["pred"], d["seg"]
                del data, test_inputs
                torch.cuda.empty_cache()
