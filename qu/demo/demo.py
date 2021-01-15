import urllib.request, urllib.parse, urllib.error
from glob import glob
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile

import requests


def get_demo_dataset():
    """If not yet present, download and expands demo dataset.

    Return path of the extracted demo dataset.
    """

    # Data folder
    data_folder = Path.home() / ".qu" / "data"

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    # Check if the demo data folder already exists
    demo_folder = data_folder / "demo"
    images_folder = demo_folder / "images"
    masks_folder = demo_folder / "masks"

    # Is the data already present?
    if demo_folder.is_dir():
        if images_folder.is_dir():
            if len(glob(images_folder / "*.tif")) == 90:
                if masks_folder.is_dir():
                    if len(glob(masks_folder / "*.tif")) == 90:
                        return demo_folder

    # Is the zip archive already present?
    archive_found = False
    if (data_folder / "demo.zip").is_file():
        if (data_folder / "demo.zip").stat().st_size > 0:
            archive_found = True

    if not archive_found:

        # Get binary stream
        r = requests.get("https://obit.ethz.ch/qu/demo.zip")

        # Target file
        with open(data_folder / "demo.zip", 'wb') as f:
            f.write(r.content)

        # Inform
        print(f"Downloaded 'demo.zip' ({(data_folder / 'demo.zip').stat().st_size} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / "demo.zip", 'r') as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder

