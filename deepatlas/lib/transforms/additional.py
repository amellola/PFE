import logging
import os

import nibabel as nib
import numpy as np
import torch
from monai.data.meta_obj import get_track_meta
from monai.transforms import Transform
from monai.transforms.transform import MapTransform
from monai.utils import convert_to_tensor

log = logging.getLogger(__name__)

# https://github.com/nvahmadi/NVIDIA_CAMP_Workshop/blob/main/01_getting_started_with_MONAI_solution.ipynb
class CleanTransform(Transform):
    def __init__(self, only_sol, skip=False) -> None:
        self.only_sol = only_sol
        self.skip = skip

    def __call__(self, data):
        if self.skip:
            return data
        if self.only_sol:
            return torch.where(data == 100, 1, 0)
        torch.where(data == 100, torch.tensor(1, dtype=data.dtype), data, out=data)
        torch.where(data == 200, torch.tensor(2, dtype=data.dtype), data, out=data)
        torch.where(data == 150, torch.tensor(3, dtype=data.dtype), data, out=data)
        torch.where(
            (data == 1) | (data == 2) | (data == 3),
            data,
            torch.tensor(0, dtype=data.dtype),
            out=data,
        )
        # if self.only_sol:
        #     return np.where(data == 100, 1, 0)
        # data = np.where(data == 100, 1, data)
        # data = np.where(data == 200, 2, data)
        # data = np.where(data == 150, 3, data)
        # data = np.where((data == 1) | (data == 2) | (data == 3), data, 0)
        # return data
        return convert_to_tensor(data, track_meta=get_track_meta())


class CleanTransformD(MapTransform):
    """Dictionary Wrapper for CleanTransform."""

    def __init__(self, keys, only_sol, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.only_sol = only_sol
        self.transform = CleanTransform(only_sol)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class SaveNib(Transform):
    def __init__(self, output_dir, dtype=None) -> None:
        self.output_dir = output_dir
        self.dtype = dtype

    def __call__(self, data):
        path = os.path.join(
            self.output_dir,
            os.path.splitext(data.meta["filename_or_obj"].split("/")[-1])[0]
            + ".nii.gz",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if len(data.shape) > 4:
            data = torch.argmax(data, dim=1).detach().cpu()[:, :, :, :]
        img = np.squeeze(data)
        converted_array = np.array(img, dtype=self.dtype if self.dtype else img.dtype)
        affine = np.eye(4)
        nifti_file = nib.Nifti1Image(converted_array, affine)
        nib.save(nifti_file, path)
        log.info("writing image to %s", path)
        return data


class SaveNibD(MapTransform):
    """Dictionary Wrapper for SaveNib."""

    def __init__(
        self, output_dir, keys, dtype=None, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.saver = SaveNib(
            output_dir=output_dir,
            dtype=dtype,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            self.saver(d[key])
        return d


class PrintFilename(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, data, prefix=""):
        name = data.meta["filename_or_obj"][0].split("/")[-1]
        log.info(f"{prefix} - transforming image {name}")
        return data


class PrintFilenameD(MapTransform):
    """Dictionary Wrapper for PrintFilename."""

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.transform = PrintFilename()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], prefix=key)
        return d


class Print(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, data, prefix=""):
        log.info("%s - transforming %s", prefix, data.split("/")[-1])
        return data


class PrintD(MapTransform):
    """Dictionary Wrapper for Print."""

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.transform = Print()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], prefix=key)
        return d
