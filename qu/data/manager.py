from enum import Enum
import gc
from glob import glob

import h5py
import numpy as np
from typing import Union, Tuple
from natsort import natsorted
from pathlib import Path
from sklearn.model_selection import train_test_split
from tifffile import imread, imsave

from qu.transform import one_hot_stack_to_label_image, label_image_to_one_hot_stack


class MaskType(Enum):
    """An enum to indicate the type of mask.

    UNKNOWN
        No supported mask type recognized.

    TIFF_LABELS
        A 2D, gray-value TIFF image. Dimensions = (1, height, width). Data type = np.int32.
        Background pixels have value = 0; all other classes (1, 2, ..., nClasses) have pixels with
        intensities matching the class number.

    NUMPY_LABELS
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

    H5_ONE_HOT
        A 3D, HDF5 array. Dimensions = (nClasses, height, width). Data type = np.float32.
        This is compatible with an ilastik export with:
            * source = "Probabilities"
            * data type = "floating 32-bit"
            * axis order = "cyx"
        Every image i = 0, ..., nClasses in the stack is associated to a class.
        Probabilities will be binarized (p > 0.5 -> True)
        Background pixels are set to True in image i = 0.
        Pixels belonging to class 1 are set to True in image i = 1.
        Pixels belonging to class 2 are set to True in image i = 2.
    """
    UNKNOWN = 0,
    TIFF_LABELS = 1,
    NUMPY_LABELS = 2,
    NUMPY_ONE_HOT = 3,
    H5_ONE_HOT = 4


class DataManager:
    """Data manager."""

    def __init__(self):
        """Constructor."""

        # Index
        self._index: int = 0

        # Number if input channels
        self._num_channels: int = 0

        # Number of classes
        self._num_classes: int = 0

        # Mask type
        self._mask_type: MaskType = MaskType.UNKNOWN

        # Paths
        self._root_data_path: str = ''
        self._rel_images_path: str = ''
        self._rel_masks_path: str = ''
        self._rel_tests_path: str = ''
        self._pred_input_path: str = ''
        self._pred_target_path: str = ''

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

        # Model path
        self._model_path = ''

        # Error message
        self._error_message = ''

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
    def num_channels(self):
        """Number of input channels."""
        if self._num_channels == 0:
            if self.num_channels > 0:
                # Force scanning
                _ = self.get_or_load_mask_at_current_index()
        return self._num_channels

    @property
    def num_classes(self):
        """Number of classes."""
        if self._num_classes == 0:
            if self.num_images > 0:
                # Force scanning
                _ = self.get_or_load_mask_at_current_index()
            else:
                raise Exception("No images found!")
        return self._num_classes

    @property
    def mask_type(self):
        """Mask type.

         @see qu.data.model.MaskType
         """
        if self._mask_type == MaskType.UNKNOWN:
            if self.num_images > 0:
                # Force scanning
                _ = self.get_or_load_mask_at_current_index()
            else:
                raise Exception("No masks found!")
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
            self._rel_tests_path = self._root_data_path / "tests"
            self._rel_runs_path = self._root_data_path / "runs"

            # If the directories do not exist, create them
            Path(self._root_data_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_images_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_masks_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_tests_path).mkdir(parents=True, exist_ok=True)
            Path(self._rel_runs_path).mkdir(parents=True, exist_ok=True)

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

    @property
    def prediction_input_path(self):
        """Prediction input path."""
        return self._pred_input_path

    @prediction_input_path.setter
    def prediction_input_path(self, value: str):
        self._pred_input_path = Path(value).resolve()

    @property
    def prediction_target_path(self):
        """Prediction target path."""
        return self._pred_target_path

    @prediction_target_path.setter
    def prediction_target_path(self, value: str):
        self._pred_target_path = Path(value).resolve()

    @property
    def model_path(self):
        """Model path."""
        return self._model_path

    @model_path.setter
    def model_path(self, value: str):
        self._model_path = Path(value).resolve()

    @property
    def last_error_message(self):
        """Return and reset last error message."""

        # Get last error message
        last_error = self._error_message

        # Reset last error message
        self._error_message = ''

        # Return
        return last_error

    def reset(self):
        """Resets all information about loaded data."""

        # Index
        self._index = 0

        # Number of classes
        self._num_classes = 0

        # Mask type
        self._mask_type = MaskType.UNKNOWN

        # Paths
        self._root_data_path = ''
        self._rel_images_path = ''
        self._rel_masks_path = ''
        self._rel_tests_path = ''

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
            return FileNotFoundError("The data root path is not set.")

        # Scan for images
        self._image_names = natsorted(
            glob(str(Path(self._rel_images_path) / "*.tif"))
        )

        # If nothing was found, return failure
        if len(self._image_names) == 0:
            raise FileNotFoundError(f"No images were found.")

        # Scan for masks

        # Reset MaskType before reassigning it
        self._mask_type = MaskType.UNKNOWN

        # First look for .tif{f} files
        self._mask_names = natsorted(
            glob(str(Path(self._rel_masks_path) / "*.tif*"))
        )

        # If no .tif{f} files were found, try with .npy files
        if len(self._mask_names) == 0:
            self._mask_names = natsorted(
                glob(str(Path(self._rel_masks_path) / "*.npy"))
            )

        # If neither .tif{f} not .npy files were found, try with .h5 files
        if len(self._mask_names) == 0:
            self._mask_names = natsorted(
                glob(str(Path(self._rel_masks_path) / "*.h5"))
            )

        # If nothing was found, return failure
        if len(self._mask_names) == 0:
            raise FileNotFoundError(f"No masks were found.")

        # Check that the number of images matches the number of masks
        if len(self._image_names) != len(self._mask_names):
            raise ValueError(f"The number of images does not match the number of masks.")

        # Return success
        return True

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

    def get_image_and_mask_at_current_index(self) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
        """Retrieve image and mask data for current index."""

        if len(self._image_names) == 0:
            return None, None

        # Get the image
        img = self.get_or_load_image_at_current_index()

        # Get the mask
        mask = self.get_or_load_mask_at_current_index()

        # Return
        return img, mask

    def get_or_load_image_at_current_index(self):
        """Get current image from cache or load it."""
        return self.get_or_load_image_at_index(self._index)

    def get_or_load_image_at_index(self, index: int):
        """Get image for requested index from cache or load it."""

        img = None

        if index in self._images:
            img = self._images[index]
        else:
            # Load it
            if index < len(self._image_names):
                image_file_name = self._image_names[index]
                if image_file_name is not None:
                    img = imread(image_file_name)

                # Add it to the cache
                self._images[index] = img

        if self._num_channels == 0:
            if img.ndim == 2:
                self._num_channels = 1
            else:
                self._num_channels = img.shape[2]

        return img

    def get_or_load_mask_at_current_index(self):
        """Get current mask from cache or load it.

        In case of failure, the last_error_message property contains
        the error message.

        @return the mask if it could be loaded properly, None otherwise.
        """
        return self.get_or_load_mask_at_index(self._index)

    def get_or_load_mask_at_index(self, index: int):
        """Get mask for requested index from cache or load it.

        In case of failure, the last_error_message property contains
        the error message.

        @param index: Index of the mask to get or load.

        @return the mask if it could be loaded properly, None otherwise.
        """

        mask = None

        if index in self._masks:
            mask = self._masks[index]
        else:
            # Load it
            if index < len(self._mask_names):
                mask_file_name = self._mask_names[index]
                if mask_file_name is not None:

                    # What is the file type?
                    extension = str(Path(mask_file_name).suffix).lower()

                    if extension == ".tif" or extension == ".tiff":

                        # Read the file
                        try:
                            mask = imread(mask_file_name)
                        except Exception as e:
                            # Set the error message
                            self._error_message = str(e)

                            # Report failure
                            return None

                        # Set the mask type
                        self._mask_type = MaskType.TIFF_LABELS

                        # Set the number of classes if not set
                        if self._num_classes == 0:
                            self._num_classes = len(np.unique(mask))

                    elif extension == ".npy":

                        # Read the file
                        try:
                            mask = np.load(mask_file_name)
                        except Exception as e:
                            # Set the error message
                            self._error_message = str(e)

                            # Report failure
                            return None

                        # Set the mask type
                        if mask.ndim == 2:
                            self._mask_type = MaskType.NUMPY_LABELS
                        else:
                            self._mask_type = MaskType.NUMPY_ONE_HOT

                        # Set the number of classes if not set
                        if self._num_classes == 0:
                            if self._mask_type == MaskType.NUMPY_LABELS:
                                self._num_classes = len(np.unique(mask))
                            else:
                                self._num_classes = mask.shape[0]

                        # Convert one-hot stack to label image for display
                        if self._mask_type == MaskType.NUMPY_ONE_HOT:
                            # Binarize the dataset
                            mask[mask > 0.5] = True

                            # Change into a labels image (for display)
                            mask = one_hot_stack_to_label_image(
                                mask,
                                first_index_is_background=True,
                                channels_first=True,
                                dtype=np.int32
                            )

                    elif extension == ".h5":

                        # Read the file
                        mask_file = h5py.File(mask_file_name, 'r')

                        # Get the dataset
                        datasets = list(mask_file.keys())
                        if len(datasets) != 1:
                            # Set the error message
                            self._error_message = f"Unexpected number of datasets in file {mask_file_name}"

                            # Report failure
                            return None

                        # Dataset name
                        dataset = datasets[0]

                        # Get the dataset
                        mask = np.array(mask_file[dataset])

                        # Close the HDF5 file
                        mask_file.close()

                        if mask.ndim == 2:
                            # Set the error message
                            self._error_message = f"The dataset in file {mask_file_name} must have "\
                                                  f"dimensions (num_classes x height x width."

                            # Report failure
                            return None

                        # Set the type
                        self._mask_type = MaskType.H5_ONE_HOT

                        # Set the number of classes if not set
                        if self._num_classes == 0:
                            self._num_classes = mask.shape[0]

                        # Binarize the dataset
                        mask[mask > 0.5] = True

                        # Change into a labels image (for display)
                        mask = one_hot_stack_to_label_image(
                            mask,
                            first_index_is_background=True,
                            channels_first=True,
                            dtype=np.int32
                        )

                    else:
                        self._mask_type = MaskType.UNKNOWN

                        # Set the error message
                        self._error_message = f"Unexpected mask file {mask_file_name}"

                        # Report failure
                        return None

                    # Make sure the mask is an int32
                    mask = mask.astype(np.int32)

                    # Add it to the cache
                    self._masks[index] = mask

        return mask

    def reload_mask_at_current_index(self):
        """Drop current mask from cache and reload it from disk."""

        # Are there any masks?
        if self.num_masks == 0:
            # Set error message
            self._error_message = "No masks loaded."

            # Report failure
            return False

        # Delete the image from cache
        if self._index in self._masks:
            del self._masks[self._index]

        # Reload
        if self.get_or_load_mask_at_current_index() is None:
            # Set error message
            self._error_message = "Could not reload mask."

            # Report failure
            return False

        # Return success
        return True

    def save_mask_at_current_index(self):
        """Save mask at current index based on the stored type.

        In case of failure, the last_error_message property contains
        the error message.

        @return True is saving was successful, False otherwise.
        """
        return self.save_mask_at_index(self._index)

    def save_mask_at_index(self, index: int):
        """Save cached mask at requested index based on the stored type.

        If the mask at given index is not in the cache, it is
        already saved to file (or was never loaded) and no operation
        is performed. The method returns success anyway, since this
        is valid behavior.

        In case of a real saving failure, the last_error_message property
        contains the error message.

        @param index: Index of the mask to save.

        @return True is saving was successful, False otherwise.
        """

        # Are there any masks?
        if self.num_masks == 0:

            # Set the error message
            self._error_message = 'No masks loaded.'

            # Return failure
            return False

        # Is the mask in the cache? If not, we stop here.
        if index not in self._masks:
            return True

        # Get the mask from cache only
        mask = self._masks[index]

        # Get the mask name
        mask_file_name = self._mask_names[index]

        # Save depending on the mask type
        if self._mask_type == MaskType.UNKNOWN:

            # Set the error message
            self._error_message = 'Unknown mask type. Cannot save.'

            # Return failure
            return False

        elif self._mask_type == MaskType.TIFF_LABELS:

            # Save as 32bit tiff
            try:
                imsave(mask_file_name, mask)
            except Exception as e:

                # Set the error message
                self._error_message = str(e)

                # Return failure
                return False

        elif self._mask_type == MaskType.NUMPY_LABELS:

            # Save 2D numpy array
            try:
                np.save(mask_file_name, mask)
            except Exception as e:

                # Set the error message
                self._error_message = str(e)

                # Return failure
                return False

        elif self._mask_type == MaskType.NUMPY_ONE_HOT:

            # Change into a one-hot stack image (for saving)
            one_hot_mask = label_image_to_one_hot_stack(
                mask,
                num_classes=self._num_classes,
                channels_first=True,
                dtype=np.int32
            )

            # Save 3D one-hot numpy array
            try:
                np.save(mask_file_name, one_hot_mask)
            except Exception as e:

                # Set the error message
                self._error_message = str(e)

                # Return failure
                return False

        elif self._mask_type == MaskType.H5_ONE_HOT:

            # Change into a one-hot stack image (for saving)
            one_hot_mask = label_image_to_one_hot_stack(
                mask,
                num_classes=self._num_classes,
                channels_first=True,
                dtype=np.int32
            )

            # Save 3D one-hot HDF5 array
            try:
                mask_file = h5py.File(mask_file_name, 'r+')
            except OSError as e:
                # Set the error message
                self._error_message = str(e)

                # Return failure
                return False

            # Get the dataset
            datasets = list(mask_file.keys())
            if len(datasets) != 1:
                # Set the error message
                self._error_message = f"Unexpected number of datasets in file {mask_file_name}"

                # Return failure
                return False

            # Dataset name
            dataset = datasets[0]

            # Replace the dataset
            mask_file[dataset][...] = one_hot_mask

            # Close the HDF5 file
            mask_file.close()

        else:

            # Set the error message
            self._error_message = f"Unknown mask type!"

            # Return failure
            return False

        # Return success
        return True

    def save_all_cached_masks(self):
        """Save all cached masks.

        The assumption is that if a mask has been loaded and
        cached, it may have been modified.

        If saving a mask fails, the process will continue saving
        the others. In this case, however, the method will return
        False, and the last error message will contain a list of
        masks that failed to save.

        @TODO: Find a way to query the mask for modifications.

        @return True if saving was successful, False otherwise.

        """

        # If there are no masks, return success.
        if self.num_masks == 0:
            return True

        failed_masks = []
        success = True
        for index in range(self.num_masks):
            if not self.save_mask_at_index(index):
                success = False
                failed_masks.append(self._mask_names[index])

        if not success:
            self._error_message = "The following masks could not be saved:\n" + \
                                  '\n'.join(map(str, failed_masks))

        return success
