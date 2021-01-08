import numpy as np
from typing import Tuple

from monai.transforms import Activations, AddChannel, AsDiscrete, \
    Compose, LoadImage, LoadNumpy, RandRotate90, RandSpatialCrop, \
    ScaleIntensity, ToTensor

from qu.ml.unet_base_learner import UNetBaseLearner
from qu.transform.transform import one_hot_stack_to_label_image


class UNetOneHotLearner(UNetBaseLearner):
    """Learner that used U-Net as worker."""

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 3,
            roi_size: Tuple[int, int] = (384, 384),
            num_epochs: int = 400,
            batch_sizes: Tuple[int, int, int] = (8, 1, 1, 1),
            num_workers: Tuple[int, int, int] = (8, 4, 8, 1),
            seed: int = 4294967295,
            working_dir: str = '.'
    ):
        """Constructor.

        @param in_channels: int, optional: default = 1
            Number of channels in the input (e.g. 1 for gray-value images).

        @param out_channels: int, optional: default = 3
            Number of channels in the output (classes).

        @param roi_size: Tuple[int, int], optional: default = (384, 384)
            Crop area (and input size of the U-Net network) used for training and validation/prediction.

        @param num_epochs: int, optional: default = 400
            Number of epochs for training.

        @param batch_sizes: Tuple[int, int, int], optional: default = (8, 1, 1, 1)
            Batch sizes for training, validation, testing, and prediction, respectively.

        @param num_workers: Tuple[int, int, int], optional: default = (4, 4, 1, 1)
            Number of workers for training, validation, testing, and prediction, respectively.

        @param seed: int
            Set random seed for modules to enable or disable deterministic training.

        @param working_dir: str, optional, default = "."
            Working folder where to save the model weights and the logs for tensorboard.

        """

        # Call base constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            roi_size=roi_size,
            num_epochs=num_epochs,
            batch_sizes=batch_sizes,
            num_workers=num_workers,
            seed=seed,
            working_dir=working_dir
        )

    def _define_transforms(self):
        """Define and initialize all data transforms.

          * training set images transform
          * training set masks transform
          * validation set images transform
          * validation set masks transform
          * validation set images post-transform
          * test set images transform
          * test set masks transform
          * test set images post-transform
          * prediction set images transform
          * prediction set images post-transform

        @return True if data transforms could be instantiated, False otherwise.
        """

        # Define transforms for training
        self._train_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                RandSpatialCrop(self._roi_size, random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                ToTensor(),
                # Informer(name="Training (image)")
            ]
        )
        self._train_mask_transforms = Compose(
            [
                LoadNumpy(data_only=True),
                RandSpatialCrop(self._roi_size, random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                ToTensor(),
                # Informer(name="Training (mask)")
            ]
        )

        # Define transforms for validation
        self._validation_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                ToTensor(),
                # Informer(name="Validation (image)")
            ]
        )
        self._validation_mask_transforms = Compose(
            [
                LoadNumpy(data_only=True),
                ToTensor(),
                # Informer(name="Validation (mask)")
            ]
        )

        # Define transforms for testing
        self._test_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                ToTensor()
            ]
        )
        self._test_mask_transforms = Compose(
            [
                LoadNumpy(data_only=True),
                ToTensor()
            ]
        )

        # Define transforms for prediction
        self._prediction_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                ToTensor()
            ]
        )

        # Post transforms
        self._validation_post_transforms = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold_values=True)
            ]
        )

        self._test_post_transforms = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold_values=True)
            ]
        )

        self._prediction_post_transforms = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold_values=True)
            ]
        )

    def _prediction_to_label_tiff_image(self, prediction):
        """Save the prediction to a label image (TIFF)"""

        # Convert to label image
        label_img = one_hot_stack_to_label_image(
            prediction,
            first_index_is_background=True,
            channels_first=True,
            dtype=np.uint16
        )

        return label_img
