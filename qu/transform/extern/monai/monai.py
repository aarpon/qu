import h5py
from monai.transforms import Transform
import numpy as np
from tifffile import imread

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
