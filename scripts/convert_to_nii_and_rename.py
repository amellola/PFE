"""Script to convert nifti files from for example mha to nii."""

import os
from pathlib import Path

import click
import SimpleITK as sitk


@click.command()
@click.option("--path", "root_path", help="Path to folder.")
@click.option("--output", "output_dir", help="Path to output folder.")
def main(root_path, output_dir):
    for dir in [f.path for f in os.scandir(root_path) if f.is_dir()]:
        filelist = Path(dir).rglob("*.mha")
        for filepath in filelist:
            path_in_str = str(filepath)
            filename = os.path.basename(path_in_str)

            mha_path = os.path.join(dir,filename)
            nii_path = os.path.join(output_dir,f'{os.path.basename(dir)}_{filename.replace(".mha", ".nii")}')

            img = sitk.ReadImage(mha_path)
            sitk.WriteImage(img, nii_path)

if __name__ == "__main__":
    main()
