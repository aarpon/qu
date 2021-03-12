import glob
from typing import Union
import numpy as np
from pathlib import Path
from tifffile import imread
from tqdm import tqdm


def find_global_intensity_range(
        images_folder: Union[Path, str],
        targets_folder: Union[Path, str] = None,
        perc: int = 0
):
    """Find the global min and max intensities of the images and target folders of images.
    
    @param: images_folder Union[Path, str]
        Image folder to scan for TIFF files. No other formats are supported for now.

    @param: targets_folder Union[None, Path, str], optional: default is None
        Target folder to scan for TIFF files. No other formats are supported for now.
        If not specified, it will be ignored.

    @param: perc float: Optional, default = 0, range = [0, 49]
        Percentile to use to find the extrema of each image.
        If perc is 0 the min and max values of the image will be returned.
        If perc is 1, the 1st and (100 - 1) 99th percentile will be returned.
        If perc is q, the qth and (100 - q)th percentile will be returned.
    """

    # Get the list of image files
    images_files = glob.glob(str(images_folder) + '/*.tif')

    # Get the list of target files (if needed)
    targets_files = []
    if targets_folder is not None:
        targets_files = glob.glob(str(targets_folder) + '/*.tif')

    # Concatenate the lists
    all_files = images_files + targets_files

    # Check that we have something
    n = len(all_files)
    if n == 0:
        return np.NaN, np.NaN

    perc = int(perc)
    if perc < 0 or perc > 49:
        raise ValueError("'perc' must be between 0 and 49.")

    # Allocate space for storing the min and max values
    min_values = np.Inf * np.ones(len(all_files))
    max_values = -np.Inf * np.ones(len(all_files))

    # Process all images      
    for i in tqdm(range(n)):
        
        # Read the image
        img = imread(all_files[i])

        # Find the min and max values at the requested percentile
        mn, mx = np.percentile(img, (perc, 100 - perc))

        # Store the results
        min_values[i] = mn
        max_values[i] = mx

    # Find the global values and return them
    return min_values.min(), max_values.max()


if __name__ == '__main__':
    folder = '/mnt/store/data/Qu/data_folder/demo_segmentation_3_classes/images/'
    mn, mx = find_global_intensity_range(folder, 1)
    print(mn, mx)
