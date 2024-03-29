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

import sys
from datetime import datetime
from glob import glob
from io import TextIOWrapper

import numpy as np
from pathlib import Path
from typing import Tuple, Union
import os

import torch
from monai.data import ArrayDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.utils import set_determinism
from monai.transforms import Activations, AddChannel, AsDiscrete, \
    Compose, LoadImage, RandRotate90, RandSpatialCrop, \
    ScaleIntensity, ToTensor
from natsort import natsorted
from tifffile import TiffWriter
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from qu.data.manager import MaskType
from qu.models.core import AttentionUNet2D, SegmentationArchitectures, SegmentationLosses, Optimizers
from qu.transform.extern.monai import ToOneHot, Identity, LoadMask
from qu.models.abstract_base_learner import AbstractBaseLearner
from qu.transform import one_hot_stack_to_label_image


class UNet2DSegmenter(AbstractBaseLearner):
    """Segmenter based on the U-Net architecture."""

    def __init__(
            self,
            architecture: SegmentationArchitectures = SegmentationArchitectures.ResidualUNet2D,
            loss: SegmentationLosses = SegmentationLosses.GeneralizedDiceLoss,
            optimizer: Optimizers = Optimizers.Adam,
            mask_type: MaskType = MaskType.TIFF_LABELS,
            in_channels: int = 1,
            out_channels: int = 3,
            roi_size: Tuple[int, int] = (384, 384),
            num_filters_in_first_layer: int = 16,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            momentum: float = 0.9,
            num_epochs: int = 400,
            batch_sizes: Tuple[int, int, int, int] = (8, 1, 1, 1),
            num_workers: Tuple[int, int, int, int] = (4, 4, 1, 1),
            validation_step: int = 2,
            sliding_window_batch_size: int = 4,
            class_names: Tuple[str, ...] = ("Background", "Object", "Border"),
            experiment_name: str = "Unet",
            model_name: str = "best_model",
            seed: int = 4294967295,
            working_dir: str = '.',
            stdout: TextIOWrapper = sys.stdout,
            stderr: TextIOWrapper = sys.stderr
    ):
        """Constructor.

        @param mask_type: MaskType
            Type of mask: defines file type, mask geometry and they way pixels
            are assigned to the various classes.

            @see qu.data.model.MaskType

        @param architecture: SegmentationArchitectures
            Core network architecture: one of (SegmentationArchitectures.ResidualUNet2D, SegmentationArchitectures.AttentionUNet2D)

        @param loss: SegmentationLosses
            Loss function: currently only SegmentationLosses.GeneralizedDiceLoss is supported

        @param optimizer: Optimizers
            Optimizer: one of (Optimizers.Adam, Optimizers.SGD)

        @param in_channels: int, optional: default = 1
            Number of channels in the input (e.g. 1 for gray-value images).

        @param out_channels: int, optional: default = 3
            Number of channels in the output (classes).

        @param roi_size: Tuple[int, int], optional: default = (384, 384)
            Crop area (and input size of the U-Net network) used for training and validation/prediction.

        @param num_filters_in_first_layer: int
            Number of filters in the first layer. Every subsequent layer doubles the number of filters.

        @param learning_rate: float, optional: default = 1e-3
            Initial learning rate for the optimizer.

        @param weight_decay: float, optional: default = 1e-4
            Weight decay of the learning rate for the optimizer.
            Used by the Adam optimizer.

        @param momentum: float, optional: default = 0.9
            Momentum of the accelerated gradient for the optimizer.
            Used by the SGD optimizer.

        @param num_epochs: int, optional: default = 400
            Number of epochs for training.

        @param batch_sizes: Tuple[int, int, int], optional: default = (8, 1, 1, 1)
            Batch sizes for training, validation, testing, and prediction, respectively.

        @param num_workers: Tuple[int, int, int], optional: default = (4, 4, 1, 1)
            Number of workers for training, validation, testing, and prediction, respectively.

        @param validation_step: int, optional: default = 2
            Number of training steps before the next validation is performed.

        @param sliding_window_batch_size: int, optional: default = 4
            Number of batches for sliding window inference during validation and prediction.

        @param class_names: Tuple[str, ...], optional: default = ("Background", "Object", "Border")
            Name of the classes for logging validation curves.

        @param experiment_name: str, optional: default = ""
            Name of the experiment that maps to the folder that contains training information (to
            be used by tensorboard). Please note, current datetime will be appended.

        @param model_name: str, optional: default = "best_model.ph"
            Name of the file that stores the best model. Please note, current datetime will be appended
            (before the extension).

        @param seed: int, optional; default = 4294967295
            Set random seed for modules to enable or disable deterministic training.

        @param working_dir: str, optional, default = "."
            Working folder where to save the model weights and the logs for tensorboard.

        """

        # Call base constructor
        super().__init__()

        # Standard pipe wrappers
        self._stdout = stdout
        self._stderr = stderr

        # Device (initialize as "cpu")
        self._device = "cpu"

        # Architecture, loss function and optimizer
        self._option_architecture = architecture
        self._option_loss = loss
        self._option_optimizer = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum

        # Mask type
        self._mask_type = mask_type

        # Input and output channels
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Define hyper parameters
        self._roi_size = roi_size
        self._num_filters_in_first_layer = num_filters_in_first_layer
        self._training_batch_size = batch_sizes[0]
        self._validation_batch_size = batch_sizes[1]
        self._test_batch_size = batch_sizes[2]
        self._prediction_batch_size = batch_sizes[3]
        self._training_num_workers = num_workers[0]
        self._validation_num_workers = num_workers[1]
        self._test_num_workers = num_workers[2]
        self._prediction_num_workers = num_workers[3]
        self._n_epochs = num_epochs
        self._validation_step = validation_step
        self._sliding_window_batch_size = sliding_window_batch_size

        # Other parameters
        self._class_names = out_channels * ["Unknown"]
        for i in range(min(out_channels, len(class_names))):
            self._class_names[i] = class_names[i]

        # Set monai seed
        set_determinism(seed=seed)

        # All file names
        self._train_image_names: list = []
        self._train_mask_names: list = []
        self._validation_image_names: list = []
        self._validation_mask_names: list = []
        self._test_image_names: list = []
        self._test_mask_names: list = []

        # Transforms
        self._train_image_transforms = None
        self._train_mask_transforms = None
        self._validation_image_transforms = None
        self._validation_mask_transforms = None
        self._test_image_transforms = None
        self._test_mask_transforms = None
        self._prediction_image_transforms = None
        self._validation_post_transforms = None
        self._test_post_transforms = None
        self._prediction_post_transforms = None

        # Datasets and data loaders
        self._train_dataset = None
        self._train_dataloader = None
        self._validation_dataset = None
        self._validation_dataloader = None
        self._test_dataset = None
        self._test_dataloader = None
        self._prediction_dataset = None
        self._prediction_dataloader = None

        # Set model architecture, loss function, metric and optimizer
        self._model = None
        self._training_loss_function = None
        self._optimizer = None
        self._validation_metric = None

        # Working directory, model file name and experiment name for Tensorboard logs.
        # The file names will be redefined at the beginning of the training.
        self._working_dir = Path(working_dir).resolve()
        self._raw_experiment_name = experiment_name
        self._raw_model_file_name = model_name

        # Keep track of the full path of the best model
        self._best_model = ''

        # Keep track of last error message
        self._message = ""

    def train(self) -> bool:
        """Run training in a separate thread (added to the global application ThreadPool)."""

        # Free memory on the GPU
        self._clear_session()

        # Check that the data is set properly
        if len(self._train_image_names) == 0 or \
                len(self._train_mask_names) == 0 or \
                len(self._validation_image_names) == 0 or \
                len(self._validation_mask_names) == 0:
            self._message = "No training/validation data found."
            return False

        if len(self._train_image_names) != len(self._train_mask_names) == 0:
            self._message = "The number of training images does not match the number of training masks."
            return False

        if len(self._validation_image_names) != len(self._validation_mask_names) == 0:
            self._message = "The number of validation images does not match the number of validation masks."
            return False

        # Define the transforms
        self._define_training_transforms()

        # Define the datasets and data loaders
        self._define_training_data_loaders()

        # Instantiate the model
        self._define_model()

        # Define the loss function
        self._define_training_loss()

        # Define the optimizer (with default parameters)
        self._define_optimizer()

        # Define the validation metric
        self._define_validation_metric()

        # Define experiment name and model name
        experiment_name, model_file_name = self._prepare_experiment_and_model_names()

        # Keep track of the best model file name
        self._best_model = model_file_name

        # Enter the main training loop
        best_metric = -1
        best_metric_epoch = -1

        epoch_loss_values = list()
        metric_values = list()

        # Initialize TensorBoard's SummaryWriter
        writer = SummaryWriter(experiment_name)

        for epoch in range(self._n_epochs):

            # Inform
            self._print_header(f"Epoch {epoch + 1}/{self._n_epochs}")

            # Switch to training mode
            self._model.train()

            epoch_loss = 0
            step = 0
            for batch_data in self._train_dataloader:

                # Update step
                step += 1

                # Get the next batch and move it to device
                inputs, labels = batch_data[0].to(self._device), batch_data[1].to(self._device)

                # Zero the gradient buffers
                self._optimizer.zero_grad()

                # Forward pass
                outputs = self._model(inputs)

                # Calculate the loss
                loss = self._training_loss_function(outputs, labels)

                # Back-propagate
                loss.backward()

                # Update weights (optimize)
                self._optimizer.step()

                # Update and store metrics
                epoch_loss += loss.item()
                epoch_len = len(self._train_dataset) / self._train_dataloader.batch_size
                if epoch_len != int(epoch_len):
                    epoch_len = int(epoch_len) + 1

                print(f"Batch {step}/{epoch_len}: train_loss = {loss.item():.4f}", file=self._stdout)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"Average loss = {epoch_loss:.4f}", file=self._stdout)
            writer.add_scalar("average_train_loss", epoch_loss, epoch + 1)

            # Validation
            if (epoch + 1) % self._validation_step == 0:

                self._print_header("Validation")

                # Switch to evaluation mode
                self._model.eval()

                # Make sure not to update the gradients
                with torch.no_grad():

                    # Global metrics
                    metric_sum = 0.0
                    metric_count = 0
                    metric = 0.0

                    # Keep track of the metrics for all classes
                    metric_sum_classes = self._out_channels * [0.0]
                    metric_count_classes = self._out_channels * [0]
                    metric_classes = self._out_channels * [0.0]

                    for val_data in self._validation_dataloader:

                        # Get the next batch and move it to device
                        val_images, val_labels = val_data[0].to(self._device), val_data[1].to(self._device)

                        # Apply sliding inference over ROI size
                        val_outputs = sliding_window_inference(
                            val_images,
                            self._roi_size,
                            self._sliding_window_batch_size,
                            self._model
                        )
                        val_outputs = self._validation_post_transforms(val_outputs)

                        # Compute overall metric
                        value, not_nans = self._validation_metric(
                            y_pred=val_outputs,
                            y=val_labels
                        )
                        not_nans = not_nans.item()
                        metric_count += not_nans
                        metric_sum += value.item() * not_nans

                        # Compute metric for each class
                        for c in range(self._out_channels):
                            value_obj, not_nans = self._validation_metric(
                                y_pred=val_outputs[:, c:c + 1],
                                y=val_labels[:, c:c + 1]
                            )
                            not_nans = not_nans.item()
                            metric_count_classes[c] += not_nans
                            metric_sum_classes[c] += value_obj.item() * not_nans

                    # Global metric
                    metric = metric_sum / metric_count
                    metric_values.append(metric)

                    # Metric per class
                    for c in range(self._out_channels):
                        metric_classes[c] = metric_sum_classes[c] / metric_count_classes[c]

                    # Print summary
                    print(f"Global metric = {metric:.4f} ", file=self._stdout)
                    for c in range(self._out_channels):
                        print(f"Class '{self._class_names[c]}' metric = {metric_classes[c]:.4f} ", file=self._stdout)

                    # Do we have the best metric so far?
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            self._model.state_dict(),
                            model_file_name
                        )
                        print(f"New best global metric = {best_metric:.4f} at epoch: {best_metric_epoch}", file=self._stdout)
                        print(f"Saved best model '{Path(model_file_name).name}'", file=self._stdout)

                    # Add validation loss and metrics to log
                    writer.add_scalar("val_mean_dice_loss", metric, epoch + 1)
                    for c in range(self._out_channels):
                        metric_name = f"val_{self._class_names[c].lower()}_metric"
                        writer.add_scalar(metric_name, metric_classes[c], epoch + 1)

        print(f"Training completed. Best_metric = {best_metric:.4f} at epoch: {best_metric_epoch}", file=self._stdout)
        writer.close()

        # Return success
        return True

    def test_predict(
            self,
            target_folder: Union[Path, str] = '',
            model_path: Union[Path, str] = ''
    ) -> bool:
        """Run prediction on predefined test data.

        @param target_folder: Path|str, optional: default = ''
            Path to the folder where to store the predicted images. If not specified,
            if defaults to '{working_dir}/predictions'. See constructor.

        @param model_path: Path|str, optional: default = ''
            Full path to the model to use. If omitted and a training was
            just run, the path to the model with the best metric is
            already stored and will be used.

            @see get_best_model_path()

        @return True if the prediction was successful, False otherwise.
        """

        # Inform
        self._print_header("Test prediction")

        # Get the device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If the model is not in memory, instantiate it first
        if self._model is None:
            self._define_model()

        # If the path to the best model was not set, use current one (if set)
        if model_path == '':
            model_path = self.get_best_model_path()

        # Try loading the model weights: they must be compatible
        # with the model in memory
        try:
            checkpoint = torch.load(
                model_path,
                map_location=torch.device('cpu')
            )
            self._model.load_state_dict(checkpoint)
            print(f"Loaded best metric model {model_path}.", file=self._stdout)
        except Exception as e:
            self._message = "Error: there was a problem loading the model! Aborting."
            return False

        # If the target folder is not specified, set it to the standard predictions out
        if target_folder == '':
            target_folder = Path(self._working_dir) / "tests"
        else:
            target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        # Switch to evaluation mode
        self._model.eval()

        indx = 0

        # Make sure not to update the gradients
        with torch.no_grad():
            for test_data in self._test_dataloader:

                # Get the next batch and move it to device
                test_images, test_masks = test_data[0].to(self._device), test_data[1].to(self._device)

                # Apply sliding inference over ROI size
                test_outputs = sliding_window_inference(
                    test_images,
                    self._roi_size,
                    self._sliding_window_batch_size,
                    self._model
                )
                test_outputs = self._test_post_transforms(test_outputs)

                # Retrieve the image from the GPU (if needed)
                pred = test_outputs.cpu().numpy().squeeze()

                # Prepare the output file name
                basename = os.path.splitext(os.path.basename(self._test_image_names[indx]))[0]
                basename = basename.replace('train_', 'pred_')

                # Convert to label image
                label_img = self._prediction_to_label_tiff_image(pred)

                # Save label image as tiff file
                label_file_name = os.path.join(
                    str(target_folder),
                    basename + '.tif')
                with TiffWriter(label_file_name) as tif:
                    tif.save(label_img)

                # Inform
                print(f"Saved {str(target_folder)}/{basename}.tif", file=self._stdout)

                # Update the index
                indx += 1

        # Inform
        print(f"Test prediction completed.", file=self._stdout)

        # Return success
        return True

    def predict(self,
                input_folder: Union[Path, str],
                target_folder: Union[Path, str],
                model_path: Union[Path, str]
                ):
        """Run prediction.

        @param input_folder: Path|str
            Path to the folder where to store the predicted images.

        @param target_folder: Path|str
            Path to the folder where to store the predicted images.

        @param model_path: Path|str
            Full path to the model to use.

        @return True if the prediction was successful, False otherwise.
        """
        # Inform
        self._print_header("Prediction")

        # Get the device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If the model is not in memory, instantiate it first
        if self._model is None:
            self._define_model()

        # Try loading the model weights: they must be compatible
        # with the model in memory
        try:
            checkpoint = torch.load(
                model_path,
                map_location=torch.device('cpu')
            )
            self._model.load_state_dict(checkpoint)
            print(f"Loaded best metric model {model_path}.", file=self._stdout)
        except Exception as e:
            self._message = "Error: there was a problem loading the model! Aborting."
            return False

        # Make sure the target folder exists
        if type(target_folder) == str and target_folder == '':
            self._message = "Error: please specify a valid target folder! Aborting."
            return False

        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        # Get prediction dataloader
        if not self._define_prediction_data_loaders(input_folder):
            self._message = "Error: could not instantiate prediction dataloader! Aborting."
            return False

        # Switch to evaluation mode
        self._model.eval()

        indx = 0

        # Make sure not to update the gradients
        with torch.no_grad():
            for prediction_data in self._prediction_dataloader:

                # Get the next batch and move it to device
                prediction_images = prediction_data.to(self._device)

                # Apply sliding inference over ROI size
                prediction_outputs = sliding_window_inference(
                    prediction_images,
                    self._roi_size,
                    self._sliding_window_batch_size,
                    self._model
                )
                prediction_outputs = self._prediction_post_transforms(prediction_outputs)

                # Retrieve the image from the GPU (if needed)
                pred = prediction_outputs.cpu().numpy().squeeze()

                # Prepare the output file name
                basename = os.path.splitext(os.path.basename(self._prediction_image_names[indx]))[0]
                basename = "pred_" + basename

                # Convert to label image
                label_img = self._prediction_to_label_tiff_image(pred)

                # Save label image as tiff file
                label_file_name = os.path.join(
                    str(target_folder),
                    basename + '.tif')
                with TiffWriter(label_file_name) as tif:
                    tif.save(label_img)

                # Inform
                print(f"Saved {str(target_folder)}/{basename}.tif", file=self._stdout)

                # Update the index
                indx += 1

        # Inform
        print(f"Prediction completed.", file=self._stdout)

        # Return success
        return True

    def set_training_data(self,
                          train_image_names,
                          train_mask_names,
                          val_image_names,
                          val_mask_names,
                          test_image_names,
                          test_mask_names) -> None:
        """Set all training files names.

        @param train_image_names: list
            List of training image names.

        @param train_mask_names: list
            List of training mask names.

        @param val_image_names: list
            List of validation image names.

        @param val_mask_names: list
            List of validation image names.

        @param test_image_names: list
            List of test image names.

        @param test_mask_names: list
            List of test image names.
        """

        # First validate all data
        if len(train_image_names) != len(train_mask_names):
            raise ValueError("The number of training images does not match the number of training masks.")

        if len(val_image_names) != len(val_mask_names):
            raise ValueError("The number of validation images does not match the number of validation masks.")

        if len(test_image_names) != len(test_mask_names):
            raise ValueError("The number of test images does not match the number of test masks.")

        # Training data
        self._train_image_names = train_image_names
        self._train_mask_names = train_mask_names

        # Validation data
        self._validation_image_names = val_image_names
        self._validation_mask_names = val_mask_names

        # Test data
        self._test_image_names = test_image_names
        self._test_mask_names = test_mask_names

    @staticmethod
    def _prediction_to_label_tiff_image(prediction):
        """Save the prediction to a label image (TIFF)"""

        # Convert to label image
        label_img = one_hot_stack_to_label_image(
            prediction,
            first_index_is_background=True,
            channels_first=True,
            dtype=np.uint16
        )

        return label_img

    def _define_training_transforms(self):
        """Define and initialize all training data transforms.

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

        if self._mask_type == MaskType.UNKNOWN:
            raise Exception("The mask type is unknown. Cannot continue!")

        # Depending on the mask type, we will need to adapt the Mask Loader
        # and Transform. We start by initializing the most common types.
        MaskLoader = LoadMask(self._mask_type)
        MaskTransform = Identity

        # Adapt the transform for the LABEL types
        if self._mask_type == MaskType.TIFF_LABELS or self._mask_type == MaskType.NUMPY_LABELS:
            MaskTransform = ToOneHot(num_classes=self._out_channels)

        # The H5_ONE_HOT type requires a different loader
        if self._mask_type == MaskType.H5_ONE_HOT:
            # MaskLoader: still missing
            raise Exception("HDF5 one-hot masks are not supported yet!")

        # Define transforms for training
        self._train_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                RandSpatialCrop(self._roi_size, random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                ToTensor()
            ]
        )
        self._train_mask_transforms = Compose(
            [
                MaskLoader,
                MaskTransform,
                RandSpatialCrop(self._roi_size, random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                ToTensor()
            ]
        )

        # Define transforms for validation
        self._validation_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                ToTensor()
            ]
        )
        self._validation_mask_transforms = Compose(
            [
                MaskLoader,
                MaskTransform,
                ToTensor()
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
                MaskLoader,
                MaskTransform,
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

    def _define_training_data_loaders(self) -> bool:
        """Initialize training datasets and data loaders.

        @Note: in Windows, it is essential to set `persistent_workers=True` in the data loaders!

        @return True if datasets and data loaders could be instantiated, False otherwise.
        """

        # Optimize arguments
        if sys.platform == 'win32':
            persistent_workers = True
            pin_memory = False
        else:
            persistent_workers = False
            pin_memory = torch.cuda.is_available()

        if len(self._train_image_names) == 0 or \
                len(self._train_mask_names) == 0 or \
                len(self._validation_image_names) == 0 or \
                len(self._validation_mask_names) == 0 or \
                len(self._test_image_names) == 0 or \
                len(self._test_mask_names) == 0:

            self._train_dataset = None
            self._train_dataloader = None
            self._validation_dataset = None
            self._validation_dataloader = None
            self._test_dataset = None
            self._test_dataloader = None

            return False

        # Training
        self._train_dataset = ArrayDataset(
            self._train_image_names,
            self._train_image_transforms,
            self._train_mask_names,
            self._train_mask_transforms
        )
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self._training_batch_size,
            shuffle=False,
            num_workers=self._training_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

        # Validation
        self._validation_dataset = ArrayDataset(
            self._validation_image_names,
            self._validation_image_transforms,
            self._validation_mask_names,
            self._validation_mask_transforms
        )
        self._validation_dataloader = DataLoader(
            self._validation_dataset,
            batch_size=self._validation_batch_size,
            shuffle=False,
            num_workers=self._validation_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

        # Test
        self._test_dataset = ArrayDataset(
            self._test_image_names,
            self._test_image_transforms,
            self._test_mask_names,
            self._test_mask_transforms
        )
        self._test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._test_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

        return True

    def _define_prediction_transforms(self):
        """Define and initialize all prediction data transforms.

          * prediction set images transform
          * prediction set images post-transform

        @return True if data transforms could be instantiated, False otherwise.
        """

        # Define transforms for prediction
        self._prediction_image_transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                AddChannel(),
                ToTensor(),
            ]
        )

        self._prediction_post_transforms = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(threshold_values=True),
            ]
        )

    def _define_prediction_data_loaders(
            self,
            prediction_folder_path: Union[Path, str]
    ) -> bool:
        """Initialize prediction datasets and data loaders.

        @Note: in Windows, it is essential to set `persistent_workers=True` in the data loaders!

        @return True if datasets and data loaders could be instantiated, False otherwise.
        """

        # Check that the path exists
        prediction_folder_path = Path(prediction_folder_path)
        if not prediction_folder_path.is_dir():
            return False

        # Scan for images
        self._prediction_image_names = natsorted(
            glob(str(Path(prediction_folder_path) / "*.tif"))
        )

        # Optimize arguments
        if sys.platform == 'win32':
            persistent_workers = True
            pin_memory = False
        else:
            persistent_workers = False
            pin_memory = torch.cuda.is_available()

        if len(self._prediction_image_names) == 0:

            self._prediction_dataset = None
            self._prediction_dataloader = None

            return False

        # Define the transforms
        self._define_prediction_transforms()

        # Prediction
        self._prediction_dataset = Dataset(
            self._prediction_image_names,
            self._prediction_image_transforms
        )
        self._prediction_dataloader = DataLoader(
            self._prediction_dataset,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._test_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

        return True

    def get_message(self):
        """Return last error message."""
        return self._message

    def get_best_model_path(self):
        """Return the full path to the best model."""
        return self._best_model

    def _clear_session(self) -> None:
        """Try clearing cache on the GPU."""
        if self._device != "cpu":
            torch.cuda.empty_cache()

    def _define_model(self) -> None:
        """Instantiate the U-Net architecture."""

        # Create U-Net
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device '{self._device}'.", file=self._stdout)

        # Try to free memory on the GPU
        if self._device != "cpu":
            torch.cuda.empty_cache()

        # Instantiate the requested model
        if self._option_architecture == SegmentationArchitectures.ResidualUNet2D:
            # Monai's UNet
            self._model = UNet(
                dimensions=2,
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                channels=tuple((self._num_filters_in_first_layer * 2**i for i in range(0, 5))),
                strides=(2, 2, 2, 2),
                num_res_units=2
            ).to(self._device)

        elif self._option_architecture == SegmentationArchitectures.AttentionUNet2D:

            # Attention U-Net
            self._model = AttentionUNet2D(
                img_ch=self._in_channels,
                output_ch=self._out_channels,
                n1=self._num_filters_in_first_layer
            ).to(self._device)

        else:
            raise ValueError(f"Unexpected architecture {self._option_architecture}! Aborting.")

    def _define_training_loss(self) -> None:
        """Define the loss function."""

        if self._option_loss == SegmentationLosses.GeneralizedDiceLoss:
            self._training_loss_function = GeneralizedDiceLoss(
                include_background=True,
                to_onehot_y=False,
                softmax=True,
                batch=True,
            )
        else:
            raise ValueError(f"Unknown loss option {self._option_loss}! Aborting.")

    def _define_optimizer(self) -> None:
        """Define the optimizer."""

        if self._model is None:
            return

        if self._option_optimizer == Optimizers.Adam:
            self._optimizer = Adam(
                self._model.parameters(),
                self._learning_rate,
                weight_decay=self._weight_decay,
                amsgrad=True
            )
        elif self._option_optimizer == Optimizers.SGD:
            self._optimizer = SGD(
                self._model.parameters(),
                lr=self._learning_rate,
                momentum=self._momentum
            )
        else:
            raise ValueError(f"Unknown optimizer option {self._option_optimizer}! Aborting.")

    def _define_validation_metric(self):
        """Define the metric for validation function."""

        self._validation_metric = DiceMetric(
            include_background=True,
            reduction="mean"
        )

    def _prepare_experiment_and_model_names(self) -> Tuple[str, str]:
        """Prepare the experiment and model names.

        @return experiment_file_name, model_file_name

        Current date time is appended and the full path is returned.
        """

        # Make sure the "runs" subfolder exists
        runs_dir = Path(self._working_dir) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y%m%d_%H%M%S")

        # Experiment name
        experiment_name = f"{self._raw_experiment_name}_{str(self._option_architecture)}_{date_time}" \
            if self._raw_experiment_name != "" \
            else f"{str(self._option_architecture)}_{date_time}"
        experiment_name = runs_dir / experiment_name

        # Best model file name
        name = Path(self._raw_model_file_name).stem
        model_file_name = f"{name}_{date_time}.pth"
        model_file_name = runs_dir / model_file_name

        return str(experiment_name), str(model_file_name)

    def _print_header(self, header_text, line_length=80, file=None):
        """Print a section header."""
        if file is None:
            file = self._stdout
        print(f"{line_length * '-'}", file=file)
        print(f"{header_text}", file=self._stdout)
        print(f"{line_length * '-'}", file=file)