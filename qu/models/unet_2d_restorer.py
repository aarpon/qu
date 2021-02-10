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

import os
import sys
from datetime import datetime
from glob import glob
from io import TextIOWrapper
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from monai.data import DataLoader, CacheDataset, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, ToTensord, ToNumpy, ScaleIntensity, LoadImage, AddChannel, ToTensor,
    ScaleIntensityd, RandSpatialCropSamplesd
)
from monai.utils import set_determinism
from natsort import natsorted
from tifffile import TiffWriter
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from qu.models.abstract_base_learner import AbstractBaseLearner
from qu.models.basic import ClassicUNet2D
from qu.transform import one_hot_stack_to_label_image
from qu.transform.extern.monai import Identity


class UNet2DRestorer(AbstractBaseLearner):
    """Restorer based on the U-Net architecture."""

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            roi_size: Tuple[int, int] = (384, 384),
            num_epochs: int = 400,
            batch_sizes: Tuple[int, int, int, int] = (8, 1, 1, 1),
            num_workers: Tuple[int, int, int, int] = (4, 4, 1, 1),
            validation_step: int = 2,
            sliding_window_batch_size: int = 4,
            experiment_name: str = "",
            model_name: str = "best_model",
            seed: int = 4294967295,
            working_dir: str = '.',
            stdout: TextIOWrapper = sys.stdout,
            stderr: TextIOWrapper = sys.stderr
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

        @param validation_step: int, optional: default = 2
            Number of training steps before the next validation is performed.

        @param sliding_window_batch_size: int, optional: default = 4
            Number of batches for sliding window inference during validation and prediction.

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

        # Input and output channels
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Define hyper parameters
        self._roi_size = roi_size
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

        # Set monai seed
        set_determinism(seed=seed)

        # All file names
        self._train_image_names: list = []
        self._train_target_names: list = []
        self._validation_image_names: list = []
        self._validation_target_names: list = []
        self._test_image_names: list = []
        self._test_target_names: list = []

        # Data dictionary
        self._train_data_dictionary = None
        self._validation_data_dictionary = None
        self._test_data_dictionary = None
        self._prediction_data_dictionary = None

        # Transforms
        self._train_transforms = None
        self._validation_transforms = None
        self._test_transforms = None

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

    def _dump_network(self, output_file_name: Union[Path, str]) -> None:
        """Dump the network structure to file."""

        if self._model is None:
            return

        # Make sure the parent folder already exists
        Path(output_file_name).parent.mkdir(parents=True, exist_ok=True)

        # Write the structure
        with open(output_file_name, 'w') as f:
            for module in self._model.modules():
                f.write(f"{module}")

    def train(self) -> bool:
        """Run training in a separate thread (added to the global application ThreadPool)."""

        # Free memory on the GPU
        self._clear_session()

        # Check that the data is set properly
        if len(self._train_data_dictionary) == 0 or \
                len(self._validation_data_dictionary) == 0:
            self._message = "No training/validation data found."
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

        # Define experiment name and model name
        experiment_name, model_file_name = self._prepare_experiment_and_model_names()

        # Keep track of the best model file name
        self._best_model = model_file_name

        # Dump the network structure
        self._dump_network(Path(experiment_name) / "UNet_architecture.txt")

        # Enter the main training loop
        lowest_validation_loss = np.Inf
        lowest_validation_epoch = -1

        epoch_loss_values = list()
        validation_loss_values = list()

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
                inputs, labels = batch_data["image"].to(self._device), batch_data["label"].to(self._device)

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

                    # Global validation loss
                    validation_loss_sum = 0.0
                    validation_loss_count = 0

                    for val_data in self._validation_dataloader:

                        # Get the next batch and move it to device
                        val_images, val_labels = val_data["image"].to(self._device), val_data["label"].to(self._device)

                        # Apply sliding inference over ROI size
                        val_outputs = sliding_window_inference(
                            val_images,
                            self._roi_size,
                            self._sliding_window_batch_size,
                            self._model
                        )
                        val_outputs = self._validation_post_transforms(val_outputs)

                        # Calculate the validation loss
                        val_loss = self._training_loss_function(val_outputs, val_labels)

                        # Add to the current loss
                        validation_loss_count += 1
                        validation_loss_sum += val_loss.item()

                    # Global validation loss
                    validation_loss = validation_loss_sum / validation_loss_count
                    validation_loss_values.append(validation_loss)

                    # Print summary
                    print(f"Validation loss = {validation_loss:.4f} ", file=self._stdout)

                    # Do we have the best metric so far?
                    if validation_loss < lowest_validation_loss:
                        lowest_validation_loss = validation_loss
                        lowest_validation_epoch = epoch + 1
                        torch.save(
                            self._model.state_dict(),
                            model_file_name
                        )
                        print(f"New lowest validation loss = {lowest_validation_loss:.4f} at epoch: {lowest_validation_epoch}", file=self._stdout)
                        print(f"Saved best model '{Path(model_file_name).name}'", file=self._stdout)

                    # Add validation loss and metrics to log
                    writer.add_scalar("val_mean_loss", validation_loss, epoch + 1)

        print(f"Training completed. Lowest validation loss = {lowest_validation_loss:.4f} at epoch: {lowest_validation_epoch}", file=self._stdout)
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

        # Make sure not to update the gradients
        with torch.no_grad():
            for indx, test_data in enumerate(self._test_dataloader):

                # Get the next batch and move it to device
                test_images, test_masks = test_data["image"].to(self._device), test_data["label"].to(self._device)

                # Apply sliding inference over ROI size
                test_outputs = sliding_window_inference(
                    test_images,
                    self._roi_size,
                    self._sliding_window_batch_size,
                    self._model
                )
                test_outputs = self._test_post_transforms(test_outputs)

                # The ToNumpy() transform already causes the Tensor
                # to be gathered from the GPU to the CPU
                pred = test_outputs.squeeze()

                # Prepare the output file name
                basename = os.path.splitext(os.path.basename(self._test_image_names[indx]))[0]
                basename = basename.replace('train_', 'pred_')

                # Save label image as tiff file
                pred_file_name = os.path.join(
                    str(target_folder),
                    basename + '.tif')
                with TiffWriter(pred_file_name) as tif:
                    tif.save(pred)

                # Inform
                print(f"Saved {str(target_folder)}/{basename}.tif", file=self._stdout)

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

                # The ToNumpy() transform already causes the Tensor
                # to be gathered from the GPU to the CPU
                pred = prediction_outputs.squeeze()

                # Prepare the output file name
                basename = os.path.splitext(os.path.basename(self._prediction_image_names[indx]))[0]
                basename = "pred_" + basename

                # Save label image as tiff file
                pred_file_name = os.path.join(
                    str(target_folder),
                    basename + '.tif')
                with TiffWriter(pred_file_name) as tif:
                    tif.save(pred)

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
        self._train_target_names = train_mask_names

        # Validation data
        self._validation_image_names = val_image_names
        self._validation_target_names = val_mask_names

        # Test data
        self._test_image_names = test_image_names
        self._test_target_names = test_mask_names

        # Training data
        self._train_data_dictionary = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_image_names, train_mask_names)
        ]

        self._validation_data_dictionary = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(val_image_names, val_mask_names)
        ]

        self._test_data_dictionary  = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_image_names, test_mask_names)
        ]

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
          * training set targets transform
          * validation set images transform
          * validation set targets transform
          * validation set images post-transform
          * test set images transform
          * test set targets transform
          * test set images post-transform
          * prediction set images transform
          * prediction set images post-transform

        @return True if data transforms could be instantiated, False otherwise.
        """
        # Define transforms for training
        self._train_transforms = Compose(
            [
                LoadImaged(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                AddChanneld(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                # ScaleIntensityRanged(
                #     keys=[
                #         "image",
                #         "label"
                #     ],
                #     a_min=0,
                #     a_max=65535,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=False
                # ),
                ScaleIntensityd(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                RandSpatialCropSamplesd(
                    keys=[
                        "image",
                        "label"
                    ],
                    roi_size=self._roi_size,
                    num_samples=16,
                    random_center=True,
                    random_size=False
                ),
                ToTensord(
                    keys=[
                        "image",
                        "label"
                    ]
                )
            ]
        )

        # Define transforms for validation
        self._validation_transforms = Compose(
            [
                LoadImaged(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                AddChanneld(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                # ScaleIntensityRanged(
                #     keys=[
                #         "image",
                #         "label"
                #     ],
                #     a_min=0,
                #     a_max=65535,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=False
                # ),
                ScaleIntensityd(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                ToTensord(
                    keys=[
                        "image",
                        "label"
                    ]
                )
            ]
        )

        # Define transforms for testing
        self._test_transforms = Compose(
            [
                LoadImaged(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                AddChanneld(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                # ScaleIntensityRanged(
                #     keys=[
                #         "image",
                #         "label"
                #     ],
                #     a_min=0,
                #     a_max=65535,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=False
                # ),
                ScaleIntensityd(
                    keys=[
                        "image",
                        "label"
                    ]
                ),
                ToTensord(
                    keys=[
                        "image",
                        "label"
                    ]
                )
            ]
        )

        # Post transforms
        self._validation_post_transforms = Compose(
            [
                Identity()
            ]
        )

        self._test_post_transforms = Compose(
            [
                ToNumpy(),
                ScaleIntensity(0, 65535),
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

        if len(self._train_data_dictionary) == 0 or \
                len(self._validation_data_dictionary) == 0 or \
                len(self._test_data_dictionary) == 0:

            self._train_dataset = None
            self._train_dataloader = None
            self._validation_dataset = None
            self._validation_dataloader = None
            self._test_dataset = None
            self._test_dataloader = None

            return False

        # Training
        # @TODO Investigate why CacheDataset fails
        # @TODO if num_workers > 1
        self._train_dataset = CacheDataset(
            data=self._train_data_dictionary,
            transform=self._train_transforms,
            cache_rate=1.0,
            num_workers=1
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
        # @TODO Investigate why CacheDataset fails
        # @TODO if num_workers > 1
        self._validation_dataset = CacheDataset(
            data=self._validation_data_dictionary,
            transform=self._validation_transforms,
            cache_rate=1.0,
            num_workers=1
        )
        self._validation_dataloader = DataLoader(
            self._validation_dataset,
            batch_size=self._validation_batch_size,
            num_workers=self._validation_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

        # Test
        # @TODO Investigate why CacheDataset fails
        # @TODO if num_workers > 1
        self._test_dataset = CacheDataset(
            data=self._test_data_dictionary,
            transform=self._test_transforms,
            cache_rate=1.0,
            num_workers=1
        )
        self._test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
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
                # ScaleIntensityRange(0, 65535, 0.0, 1.0, clip=False),
                ScaleIntensity(),
                AddChannel(),
                ToTensor()
            ]
        )

        self._prediction_post_transforms = Compose(
            [
                ToNumpy(),
                ScaleIntensity(0, 65535)
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
            batch_size=self._prediction_batch_size,
            shuffle=False,
            num_workers=self._prediction_num_workers,
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

        # Classic U-Net from Ronneberger et al. (with slightly different parameters)
        self._model = ClassicUNet2D(
            in_channels=self._in_channels,
            n_classes=self._out_channels,
            depth=5,
            wf=4,
            padding=True,
            batch_norm=False
        ).to(self._device)

    def _define_training_loss(self) -> None:
        """Define the loss function."""

        # Use the MAE loss
        self._training_loss_function = L1Loss()

    def _define_optimizer(
            self,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4
    ) -> None:
        """Define the optimizer.

        @param learning_rate: float, optional, default = 1e-3
            Initial learning rate for the optimizer.

        @param weight_decay: float, optional, default = 1e-4
            Weight decay of the learning rate for the optimizer.

        """

        if self._model is None:
            return

        self._optimizer = Adam(
            self._model.parameters(),
            learning_rate,
            weight_decay=weight_decay,
            amsgrad=True
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
        experiment_name = f"{self._raw_experiment_name}_{date_time}" \
            if self._raw_experiment_name != "" \
            else f"{date_time}"
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