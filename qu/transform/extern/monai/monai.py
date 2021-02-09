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

import h5py
from monai.transforms import Transform
import numpy as np
from tifffile import imread
import torch

from qu.data.manager import MaskType
from qu.transform.transform import label_image_to_one_hot_stack


class ToOneHot(Transform):
    """Converts a label image into a stack of binary images (one-hot).

    The binary image at index 'i' in the stack contains the pixels from the
    class i in the label image.

    It is recommended to set the background to class 0 in the label image.

    The OneHot stack is in the form [CxHxW]

    Args:

        num_classes: int
            Expected number of classes. If omitted, the number of classes will be
            extracted from the image. When processing a series of images, it is
            best to explicitly set num_classes, otherwise some image stacks may
            have less planes if a class happens to be missing from those images.
    """

    def __init__(self, num_classes: int) -> None:
        """Constructor"""
        self.num_classes = num_classes

    def __call__(self, label: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `label`.

        @return a stack of images with the same width and height as `label` and
            with `num_classes` planes.
        """

        # Make sure 'label' is a numpy array
        label = np.array(label)

        return label_image_to_one_hot_stack(
            label,
            num_classes=self.num_classes,
            channels_first=True,
            dtype=np.float32
        )


class Identity(Transform):
    """Returns the input untouched.

    Use this when you need to specify a Transform, but no operation should be
    applied to the input tensor.
    """

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Just return the input.

        @return the input, untouched.
        """

        return input


class LoadMask(Transform):
    """Loads all types of masks supported by Qu."""

    def __init__(self, mask_type: MaskType):

        # Store the mask type
        self._mask_type = mask_type

    def __call__(self, mask_file_name):
        """Call the transform."""

        if self._mask_type == MaskType.TIFF_LABELS:
            mask = imread(mask_file_name)
        elif self._mask_type == MaskType.NUMPY_LABELS or self._mask_type == MaskType.NUMPY_ONE_HOT:
            mask = np.load(mask_file_name)
        elif self._mask_type == MaskType.H5_ONE_HOT:
            mask_file = h5py.File(mask_file_name, 'r')
            datasets = list(mask_file.keys())
            if len(datasets) != 1:
                raise Exception(f"Unexpected number of datasets in file {mask_file_name}")
            dataset = datasets[0]
            mask = np.array(mask_file[dataset])
            mask_file.close()
        else:
            raise Exception(f"Unknown type for mask {mask_file_name}")

        mask = mask.astype(np.int32)

        return mask


class DebugInformer(Transform):
    """
    Simple reporter to be added to a Composed list of Transforms
    to return some information. The data is returned untouched.
    """
    def __init__(self, *args, **kwargs):
        """Constructor.

        @param name: str
            Name of the Informer. It is printed as a prefix to the output.
        """
        super().__init__()
        self.name = ""
        if "name" in kwargs:
            self.name = kwargs['name']

    def __call__(self, data):
        """Call the Transform."""
        prefix = f"{self.name} :: " if self.name != "" else ""
        if type(data) == torch.Tensor:
            print(
                f"{prefix}"
                f"Type = Torch Tensor: "
                f"size = {data.size()}, "
                f"type = {data.dtype}, "
                f"min = {data.min()}, "
                f"mean = {data.mean()}, "
                f"median = {torch.median(data).item()}, "
                f"max = {data.max()}"
            )
        elif type(data) == np.ndarray:
            print(
                f"{prefix}"
                f"Type = Numpy Array: "
                f"size = {data.shape}, "
                f"type = {data.dtype}, "
                f"min = {data.min()}, "
                f"mean = {data.mean()}, "
                f"median = {np.median(data)}, "
                f"max = {data.max()}"
            )
        elif type(data) == str:
            print(f"{prefix}String: value = '{str}'")
        elif str(type(data)).startswith("<class 'itk."):
            # This is a bit of a hack..."
            data_numpy = np.array(data)
            print(
                f"{prefix}"
                f"Type = ITK Image: "
                f"size = {data_numpy.shape}, "
                f"type = {data_numpy.dtype}, "
                f"min = {data_numpy.min()}, "
                f"mean = {data_numpy.mean()}, "
                f"median = {np.median(data_numpy)}, "
                f"max = {data_numpy.max()}"
            )
        else:
            try:
                print(f"{prefix}{type(data)}: {str(data)}")
            except:
                print(f"{prefix}Unknown type!")
        return data
