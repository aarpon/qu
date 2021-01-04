from enum import Enum
import gc
from glob import glob
import numpy as np
from typing import Union, Tuple
from natsort import natsorted
from pathlib import Path
from sklearn.model_selection import train_test_split
from tifffile import imread


class MaskType(Enum):
    """An enum to indicate the type of mask.

    TIFF_LABELS
        A 2D, gray-value TIFF image. Dimensions = (1, height, width). Data type = np.int32.
        Background pixels have value = 0; all other classes (1, 2, ..., nClasses) have pixels with
        intensities matching the class number.

    TIFF_LABELS
        A 2D numpy array. Dimensions = (1, height, width). Data type = np.int32.
        Background pixels have value = 0; all other classes (1, 2, ..., nClasses) have pixels with
        intensities matching the class number.

    NUMPY_ONE_HOT
        A 3D, binary numpy array. Dimensions = (nClasses, height, width). Data type = np.int32.
        Every image i = 0, ..., nClasses in the stack is associated to a class.
        Background pixels are set to True in image i = 0.
        Pixels belonging to class 1 are set to True in image i = 1.
        Pixels belonging to class 2 are set to True in image i = 2.
        ...
    """
    TIFF_LABELS = 0,
    NUMPY_LABELS = 1,
    NUMPY_ONE_HOT = 2


class DataModel:
    """Data model."""

    def __init__(self):
        """Constructor."""

        # Index
        self._index: int = 0

        # Number of classes
        self._num_classes: int = 0

        # Mask type
        self._mask_type: Union[None, MaskType] = None

        # Paths
        self._root_data_path: str = ''
        self._rel_images_path: str = ''
        self._rel_masks_path: str = ''
        self._rel_predictions_path: str = ''

        # File names
        self._image_names: list = []
        self._mask_names: list = []

        # Caches
        self._images: dict = {}
        self._masks: dict = {}

        # Training
        self._training_image_names: list = []
        self._training_mask_names: list = []
        self._validation_image_names: list = []
        self._validation_mask_names: list = []
        self._test_image_names: list = []
        self._test_mask_names: list = []

        # Splits (values between 0.00 and 1.00)
        self._training_fraction: float = 0.75
        self._validation_fraction: float = 0.80

    @property
    def index(self):
        """Current image/mask index."""
        return self._index

    @index.setter
    def index(self, value: int):

        """Set current image/index."""
        if len(self._image_names) == 0:
            return

        if value > len(self._image_names) - 1:
            raise ValueError(f"The index {value} is out of bounds.")

        # Update the index
        self._index = value

    @property
    def num_images(self):
        """Number of images."""
        return len(self._image_names)

    @property
    def num_masks(self):
        """Number of masks."""
        return len(self._mask_names)

    @property
    def num_classes(self):
        """Number of classes."""
        if self._num_classes == 0:
            if self.num_images > 0:
                # Force scanning
                _ = self._get_or_load_mask()
        return self._num_classes

    @property
    def mask_type(self):
        """Mask type.

         @see qu.data.model.MaskType
         """
        if self._mask_type is None:
            if self.num_images > 0:
                # Force scanning
                _ = self._get_or_load_mask()
        return self._mask_type

    @property
    def root_data_path(self):
        """Path to the root of the data directory."""
        return self._root_data_path

    @root_data_path.setter
    def root_data_path(self, value: Union[Path, str]):

        """Set the root data path."""
        if value != '':

            # Set all paths
            self._root_data_path = Path(value).resolve()
            self._rel_images_path = self._root_data_path / "images"
            self._rel_masks_path = self._root_data_path / "masks"
            self._rel_predictions_path = self._root_data_path / "predictions"

            # If the directories do not exist, create them
            Path(self._root_data_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_images_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_masks_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_predictions_path).mkdir(parents=True, exist_ok=True)

    @property
    def training_fraction(self):
        """Fraction of training images."""
        return self._training_fraction

    @training_fraction.setter
    def training_fraction(self, value: float = 0.75):
        if value < 0.0 or value > 1.0:
            raise ValueError("The value of 'training_factor' must be 0 <= value <= 1.")
        self._training_fraction = value

    @property
    def validation_fraction(self):
        """Fraction of training images."""
        return self._validation_fraction

    @validation_fraction.setter
    def validation_fraction(self, value: float = 0.80):
        if value < 0.0 or value > 1.0:
            raise ValueError("The value of 'validation_fraction' must be 0 <= value <= 1.")
        self._validation_fraction = value

    def reset(self):
        """Resets all information about loaded data."""

        # Index
        self._index = 0

        # Number of classes
        self._num_classes = 0

        # Mask type
        self._mask_type = None

        # Paths
        self._root_data_path = ''
        self._rel_images_path = ''
        self._rel_masks_path = ''
        self._rel_predictions_path = ''

        # File names
        self._image_names = []
        self._mask_names = []

        # Caches
        self._images = {}
        self._masks = {}

        # Training
        self._training_image_names = []
        self._training_mask_names = []
        self._validation_image_names = []
        self._validation_mask_names = []
        self._test_image_names = []
        self._test_mask_names = []

        # Force garbage collection
        gc.collect()

    def scan(self):
        """Scan the data paths."""

        # If the root data path is not set, return
        if self._root_data_path == '':
            return

        # Scan for images
        self._image_names = natsorted(
            glob(str(Path(self._rel_images_path) / "*.tif"))
        )

        # First look for .npy files
        self._mask_names = natsorted(
            glob(str(Path(self._rel_masks_path) / "*.npy"))
        )

        # If no .npy files were found, try with .tif{f} files
        if len(self._mask_names) == 0:
            self._mask_names = natsorted(
                glob(str(Path(self._rel_masks_path) / "*.tif*"))
            )

    def preview_training_split(self):
        """Pre-calculate the number of images in the training, validation, and testing split.

        @return tuple with number of images in training set, number of images
            in the validation set, and number of images in the test set.

        Example:
            * total number of images = 100
            * training_fraction = 0.6, validation_fraction = 0.8
            * num_training_images = 0.6 * 100 = 60
            * num_validation_images = 0.8 * (1 - 0.6) * 100 = 32
            * num_test_images = (1.0 - 0.8) * (1 - 0.6) * 100 = 8
        """

        # Total number of images
        total_num = len(self._image_names)
        if total_num == 0:
            return 0, 0, 0

        relative_validation_fraction = 1.0 - self._training_fraction
        relative_test_fraction = 1.0 - self._validation_fraction

        num_training_images = int(
            self._training_fraction * total_num
        )
        num_validation_images = int(
            self._validation_fraction * relative_validation_fraction * total_num
        )
        num_test_images = int(
            relative_test_fraction * relative_validation_fraction * total_num
        )

        if num_validation_images < 1:
            num_validation_images = 1

        # Final validation
        d = total_num - (
                num_training_images + num_validation_images + num_test_images
        )
        if d > 0:
            num_test_images += d

        # Return
        return num_training_images, num_validation_images, num_test_images

    def training_split(self):
        """Split the image and mask names into a training, validation and test sets.

        @return Tuple with:
            list of training images,
            list of training masks,
            list of validation images,
            list of validation masks,
            list of test images,
            list of test masks
        """

        # Were the images already scanned?
        if len(self._image_names) == 0:
            self.scan()

        # Make sure that the scanning really worked.
        if len(self._image_names) == 0:
            return [], [], [], [], [], []

        # Training set
        train_image_names, rest_image_names, train_mask_names, rest_mask_names = train_test_split(
            self._image_names,
            self._mask_names,
            test_size=(1.00 - self._training_fraction),
            random_state=0,
            shuffle=True
        )

        # Validation and test sets
        val_image_names, test_image_names, val_mask_names, test_mask_names = train_test_split(
            rest_image_names,
            rest_mask_names,
            test_size=(1.00 - self._validation_fraction),
            random_state=0,
            shuffle=False
        )

        # Return
        return train_image_names, train_mask_names, val_image_names, val_mask_names, test_image_names, test_mask_names

    def get_data_for_current_index(self) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
        """Retrieve image and mask data for current index."""

        if len(self._image_names) == 0:
            return None, None

        # Get the image
        img = self._get_or_load_image()

        # Get the mask
        mask = self._get_or_load_mask()

        # Return
        return img, mask

    def _get_or_load_image(self):
        """Get current image from cache or load it."""

        img = None

        if self._index in self._images:
            img = self._images[self._index]
        else:
            # Load it
            if self._index < len(self._image_names):
                image_file_name = self._image_names[self._index]
                if image_file_name is not None:
                    img = imread(image_file_name)

                # Add it to the cache
                self._images[self._index] = img

        return img

    def _get_or_load_mask(self):
        """Get current mask from cache or load it."""

        mask = None

        if self._index in self._masks:
            mask = self._masks[self._index]
        else:
            # Load it
            if self._index < len(self._mask_names):
                mask_file_name = self._mask_names[self._index]
                if mask_file_name is not None:

                    # Do we have an npy file or a tiff file?
                    if str(mask_file_name).endswith(".tif") or \
                            str(mask_file_name).endswith(".tiff"):

                        # Read the file
                        mask = imread(mask_file_name)

                        # Set the mask type if not set
                        if self._mask_type is None:
                            self._mask_type = MaskType.TIFF_LABELS

                        # Set the number of classes if not set
                        if self._num_classes == 0:
                            self._num_classes = len(np.unique(mask))

                    elif str(mask_file_name).endswith(".npy"):

                        # Read the file
                        mask = np.load(mask_file_name)

                        # Set the mask type
                        if self._mask_type is None:
                            if mask.ndim == 2:

                                # Set the type
                                self._mask_type = MaskType.NUMPY_LABELS

                                # Set the number of classes if not set
                                if self._num_classes == 0:
                                    self._num_classes = len(np.unique(mask))

                            else:

                                # Set the type
                                self._mask_type = MaskType.NUMPY_ONE_HOT

                                # Set the number of classes if not set
                                if self._num_classes == 0:
                                    self._num_classes = mask.shape[0]

                    else:
                        raise Exception(f"Unexpected mask file {mask_file_name}")

                    # Make sure the mask is an int32
                    mask = mask.astype(np.int32)

                    # Add it to the cache
                    self._masks[self._index] = mask

        return mask
