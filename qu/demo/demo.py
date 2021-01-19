import urllib.request, urllib.parse, urllib.error
from glob import glob
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile

import requests


def get_demo_segmentation_dataset():
    """If not yet present, download and expands segmentation demo dataset.

    Return path of the extracted segmentation demo dataset.
    """

    # Data folder
    data_folder = Path.home() / ".qu" / "data"

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    # Check if the demo data folder already exists
    demo_folder = data_folder / "demo_segmentation"
    images_folder = demo_folder / "images"
    masks_folder = demo_folder / "masks"

    # Is the data already present?
    if demo_folder.is_dir():
        if images_folder.is_dir():
            if len(glob(str(images_folder / "*.tif*"))) == 90:
                if masks_folder.is_dir():
                    if len(glob(str(masks_folder / "*.tif*"))) == 90:
                        return demo_folder

    # Is the zip archive already present?
    archive_found = False
    if (data_folder / "demo_segmentation.zip").is_file():
        if (data_folder / "demo_segmentation.zip").stat().st_size == 81409292:
            archive_found = True
        else:
            (data_folder / "demo_segmentation.zip").unlink()

    if not archive_found:

        # Get binary stream
        r = requests.get("https://obit.ethz.ch/qu/demo_segmentation.zip")

        # Target file
        with open(data_folder / "demo_segmentation.zip", 'wb') as f:
            f.write(r.content)

        # Inform
        print(f"Downloaded 'demo_segmentation.zip' ({(data_folder / 'demo_segmentation.zip').stat().st_size} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / "demo_segmentation.zip", 'r') as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder

