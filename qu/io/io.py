#   /********************************************************************************
#   * Copyright © 2020-2021, ETH Zurich, D-BSSE, Aaron Ponti
#   * All rights reserved. This program and the accompanying materials
#   * are made available under the terms of the Apache License Version 2.0
#   * which accompanies this distribution, and is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.txt
#   *
#   * Contributors:
#   *     Aaron Ponti - initial API and implementation
#   *******************************************************************************/
#
from pathlib import Path
from typing import Union, Tuple
from tifffile import imread, imsave
import glob
from natsort import natsorted
import numpy as np


def create_blocks(
        input_tiff_file: Union[Path, str],
        output_dir: Union[None, Path, str] = None,
        block_size: Tuple[int, int] = (512, 512),
        verbose: bool = False) -> bool:
    """Breaks a 2D or 3D data set into blocks of size (height, widht) and full depth.

    @param input_tiff_file: Path or str
        Full path of the TIFF file to read.

    @param output_dir: None, Path or str, default = None
        Full path to the target directory. Set to None (or omit), to save in the folder of the source TIFF file.

    @param block_size: Tuple(height, width), default = (512, 512)
        Size of the block (height, width); all other dimensions are preserved completely.

    @param verbose: Bool
        If True, print a lot of information.

    Blocks are saved with filenames following this pattern: {work_dir} / {basename}_c####_r####.tif

    @return True if the process was successful, false otherwise.
    """

    # Try opening the file; if it fails, return False
    try:
        img = imread(input_tiff_file)
    except:
        print(f"Could not read file {input_tiff_file}. Aborting.")
        return False

    # Prepare path and file name information
    if output_dir is not None:
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(input_tiff_file).parent
    basename = Path(input_tiff_file).stem

    # Image size
    image_height = img.shape[-2]
    image_width = img.shape[-1]

    # Step sizes
    step_y = block_size[0] // 2
    step_x = block_size[1] // 2

    # Define the operation grid
    centers_y = range(step_y, image_height, step_y)
    centers_x = range(step_x, image_width, step_x)

    # Process the image row first
    for c, center_y in enumerate(centers_y):
        for r, center_x in enumerate(centers_x):

            # Inform
            if verbose:
                print(f"Extract region y = {center_y - step_y}:{center_y + step_y}, x = {center_x - step_x}:{center_x + step_x}")

            # Extract current block
            block = img[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): (center_x + step_x)]

            # Build file name
            out_file_name = work_dir / f"{basename}_c{c:04d}_r{r:04d}.tif"

            # Save the block
            imsave(str(out_file_name), block)

            # Inform
            print(f"Saved file {out_file_name.stem}")

    return True


def merge_blocks(
        input_dir: Union[Path, str],
        basename: str,
        output_dir: Union[None, Path, str] = None,
        squeeze: bool = False,
        verbose: bool = False
) -> bool:
    """Merge 2D or 3D blocks into a full image.

    @param input_dir: Path or str
        Full path of the folder where the files to be merged are contained.

    @param basename: str
        Common base name of all block files (see below).

    @param output_dir: None, Path or str, default = None
        Full path to the target directory. Set to None (or omit), to save in the folder of the source TIFF file.

    @param squeeze: Bool
        If True, squeeze any singleton dimension.

    @param verbose: Bool
        If True, print a lot of information.

    Files (blocks) are expected to have filenames following this pattern: {input_dir} / {basename}_c####_r####.tif

    @return True if the process was successful, false otherwise.    
    """

    # Try to get the list of files to merge
    file_list = natsorted(glob.glob(str(Path(input_dir) / f"{basename}_c????_r????.tif")))
    if len(file_list) == 0:
        print("No files found. Aborting.")
        return False

    # Make sure output_dir is defined and exists
    if output_dir is not None:
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(input_dir)

    # Full path of the first image
    first_block_file_name = Path(input_dir) / f"{basename}_c0000_r0000.tif"

    # Inform
    print(f"Load and process block image {first_block_file_name}")

    # Read the first image to get the size of the individual block
    block = imread(first_block_file_name)

    # Image size
    block_height = block.shape[-2]
    block_width = block.shape[-1]

    # Step sizes
    step_y = block_height // 2
    step_x = block_width // 2

    # Get the numbers of blocks per row and column by scanning the file names
    last_row = -1
    last_col = -1
    for file_name in file_list:
        c = int(file_name[-14:-10])
        if c > last_col:
            last_col = c
        r = int(file_name[-8:-4])
        if r > last_row:
            last_row = r

    # Calculate the final image size
    image_height = step_y * (last_col + 2)
    image_width = step_x * (last_row + 2)

    # Inform
    if verbose:
        print(f"Block size = ({block_height}, {block_width}); "
              f"grid size (with overlap) = ({last_row}, {last_col}), "
              f"final image size = ({image_height}, {image_width})")

    # Allocate memory for the final image
    image_size = list(block.shape)
    image_size[-2] = image_height
    image_size[-1] = image_width
    image_dtype = block.dtype
    image = np.zeros(tuple(image_size), dtype=image_dtype)

    # Define the operation grid
    centers_y = range(step_y, image_height, step_y)
    centers_x = range(step_x, image_width, step_x)

    # Process the image row first
    for c, center_y in enumerate(centers_y):
        for r, center_x in enumerate(centers_x):

            # Do not read the very first block, since we already did
            if c == 0 and r == 0:

                # Inform
                if verbose:
                    print(f"First block already loaded")

            else:

                # Build the expected block file name
                block_file_name = Path(input_dir) / f"{basename}_c{c:04d}_r{r:04d}.tif"

                # Inform
                print(f"Load and process block image {block_file_name}")

                # Load the block
                block = imread(block_file_name)

            # Inform
            if verbose:
                print(f"Insert block into region y = "
                      f"{center_y - step_y}:{center_y + step_y}, "
                      f"x = {center_x - step_x}:{center_x + step_x}"
                      )

            # Insert current block; averaging the overlapping areas where needed
            if c == 0 and r == 0:
                image[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): (center_x + step_x)] = block

            elif c == 0 and r > 0:
                # Only average on the left

                if verbose:
                    print(f"Reading area [{center_y - step_y}:{center_y + step_y}, {center_x - step_x}:{center_x}] from image.")
                    print(f"Reading area [0:{block_height}, 0:{step_x}] from block.")
                    print(f"Storing averaged overlapping area to [0:{block_height}, 0:{step_x}] in block.")
                    print(f"Storing modified block in area to [{center_y - step_y}:{center_y + step_y}, {center_x - step_x}:{center_x + step_x}] in image.")

                overlap_image = image[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): center_x].astype(np.float32)
                working_block = block.astype(np.float32)
                working_block[..., 0:block_height, 0: step_x] = (0.5 * (working_block[..., 0: block_height, 0: step_x] + overlap_image)).astype(image_dtype)
                image[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): (center_x + step_x)] = working_block


            elif c > 0 and r == 0:
                # Only average at the top

                if verbose:
                    print(f"Reading area [{center_y - step_y}:{center_y}, {center_x - step_x}:{center_x + step_x}] from image.")
                    print(f"Reading area [0:{step_y}, 0:{block_width}] from block.")
                    print(f"Storing averaged overlapping area to [0:{block_height}, 0:{step_x}] in block.")
                    print(f"Storing modified block in area to [{center_y - step_y}:{center_y + step_y}, {center_x - step_x}:{center_x + step_x}] in image.")

                overlap_image = image[..., (center_y - step_y): (center_y), (center_x - step_x): center_x + step_x].astype(np.float32)
                working_block = block.astype(np.float32)
                working_block[..., 0:step_y, 0: block_width] = (0.5 * (working_block[..., 0:step_y, 0: block_width] + overlap_image)).astype(image_dtype)
                image[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): (center_x + step_x)] = working_block

            else:
                # Average both left and up (three segments)

                if verbose:
                    print(f"Reading area 1 [{center_y - step_y}:{center_y}, {center_x - step_x}:{center_x}] from image.")
                    print(f"Reading area 2 [{center_y - step_y}:{center_y}, {center_x}:{center_x + step_x}] from image.")
                    print(f"Reading area 3 [{center_y}:{center_y + step_y}, {center_x - step_x}:{center_x}] from image.")

                    print(f"Reading area 1 [0:{step_y}, 0:{step_x}] from block.")
                    print(f"Reading area 2 [0:{step_y}, {step_x}:{block_width}] from block.")
                    print(f"Reading area 3 [{step_y}:{block_height}, 0:{step_x}] from block.")

                    print(f"Averaging areas 1 to [0:{step_y}, 0:{step_x}] in block.")
                    print(f"Averaging areas 2 to [0:{step_y}, {step_x}:{block_width}] in block.")
                    print(f"Averaging areas 3 to [{step_y}:{block_height}, 0:{step_x}] in block.")

                    print(f"Storing modified block in area to [{center_y - step_y}:{center_y + step_y}, {center_x - step_x}:{center_x + step_x}] in image.")

                overlap_image_area_1 = image[..., (center_y - step_y): center_y, (center_x - step_x): center_x].astype(np.float32)
                overlap_image_area_2 = image[..., (center_y - step_y): center_y, center_x: (center_x + step_x)].astype(np.float32)
                overlap_image_area_3 = image[..., center_y: center_y + step_y, (center_x - step_x): center_x].astype(np.float32)
                working_block = block.astype(np.float32)
                working_block[..., 0:step_y, 0:step_x] = (0.5 * (working_block[..., 0:step_y, 0: step_x] + overlap_image_area_1)).astype(image_dtype)
                working_block[..., 0:step_y, step_x:block_width] = (0.5 * (working_block[..., 0:step_y, step_x:block_width] + overlap_image_area_2)).astype(image_dtype)
                working_block[..., step_y:block_height, 0:step_x] = (0.5 * (working_block[..., step_y:block_height, 0: step_x] + overlap_image_area_3)).astype(image_dtype)

                image[..., (center_y - step_y): (center_y + step_y), (center_x - step_x): (center_x + step_x)] = working_block

    # If squeeze is True, remove any singleton dimension
    if squeeze:
        image = image.squeeze()

    # Save the reconstructed image
    out_file_name = Path(output_dir) / f"{basename}.tif"
    imsave(out_file_name, image)

    # Return success
    return True

