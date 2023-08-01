"""Script to convert nifti files from for example mha to nii."""

import os
from pathlib import Path

import click
import SimpleITK as sitk


@click.command()
@click.option("--path", "root_path", help="Path to folder.")
def main(root_path):

    filelist = Path(root_path).rglob("*.mha")
    for filepath in filelist:
        path_in_str = str(filepath)
        filename = os.path.basename(path_in_str)

        mha_path = root_path + filename
        nii_path = root_path + filename.replace(".mha", ".nii")

        img = sitk.ReadImage(mha_path)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, nii_path)


if __name__ == "__main__":
    main()
