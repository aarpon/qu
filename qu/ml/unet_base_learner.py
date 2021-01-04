import sys
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Tuple
import os

import torch
from monai.data import ArrayDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.utils import set_determinism
from tifffile import TiffWriter
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


class UNetBaseLearner:
    """Learner that used U-Net as worker."""

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 3,
            roi_size: Tuple[int, int] = (384, 384),
            num_epochs: int = 400,
            batch_sizes: Tuple[int, int, int] = (8, 1, 1),
            num_workers: Tuple[int, int, int] = (4, 4, 1),
            validation_interval: int = 2,
            sliding_window_batch_size: int = 4,
            class_names: Tuple[str, ...] = ("Background", "Object", "Border"),
            experiment_name: str = "",
            model_name: str = "best_model",
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

        @param batch_sizes: Tuple[int, int, int], optional: default = (8, 1, 1)
            Batch sizes for training, validation and testing, respectively.

        @param num_workers: Tuple[int, int, int], optional: default = (8, 8, 8)
            Number of workers for training, validation and testing, respectively.

        @param validation_interval: int, optional: default = 2
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
        self._training_num_workers = num_workers[0]
        self._validation_num_workers = num_workers[1]
        self._test_num_workers = num_workers[2]
        self._n_epochs = num_epochs
        self._validation_interval = validation_interval
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
        self._validation_post_transforms = None
        self._test_post_transforms = None

        # Datasets and data loaders
        self._train_dataset = None
        self._train_dataloader = None
        self._validation_dataset = None
        self._validation_dataloader = None
        self._test_dataset = None
        self._test_dataloader = None

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
        self._define_transforms()

        # Define the datasets and data loaders
        self._define_data_loaders()

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
            print(f"{40 * '-'}")
            print(f"Epoch {epoch + 1}/{self._n_epochs}")
            print(f"{40 * '-'}")

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
                epoch_len = len(self._train_dataset) // self._train_dataloader.batch_size

                print(f"Batch {step}/{epoch_len}: train_loss = {loss.item():.4f}")

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"Average loss = {epoch_loss:.4f}")
            writer.add_scalar("average_train_loss", epoch_loss, epoch + 1)

            # Validation
            if (epoch + 1) % self._validation_interval == 0:

                print(f"{40 * '-'}")
                print(f"Validation")
                print(f"{40 * '-'}")

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
                    print(f"Global metric = {metric:.4f} ")
                    for c in range(self._out_channels):
                        print(f"Class '{self._class_names[c]}' metric = {metric_classes[c]:.4f} ")

                    # Do we have the best metric so far?
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            self._model.state_dict(),
                            model_file_name
                        )
                        print(f"New best global metric = {best_metric:.4f} at epoch: {best_metric_epoch}")
                        print(f"Saved best model '{Path(model_file_name).name}'")

                    # Add validation loss and metrics to log
                    writer.add_scalar("val_mean_dice_loss", metric, epoch + 1)
                    for c in range(self._out_channels):
                        metric_name = f"val_{self._class_names[c].lower()}_metric"
                        writer.add_scalar(metric_name, metric_classes[c], epoch + 1)

        print(f"Training completed. Best_metric = {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()

        # Return success
        return True

    def test_predict(self, prediction_output_path: str = '', best_model_path: str = '') -> bool:
        """Run prediction.

        @param prediction_output_path: str, optional: default = ''
            Path to the folder where to store the predicted images. If not specified,
            if defaults to '{working_dir}/predictions'. See constructor.

        @param best_model_path: str, optional: default = ''
            Full path to the model to use. If omitted and a training was
            just run, the path to the model with the best metric is
            already stored and will be used.

            @see get_best_model_path()

        @return True if the prediction was successful, False otherwise.
        """

        # Inform
        print(f"{40 * '-'}")
        print(f"Test prediction")
        print(f"{40 * '-'}")

        # Get the device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If the model is not in memory, instantiate it first
        if self._model is None:
            self._define_model()

        # If the path to the best model was not set, use current one (if set)
        if best_model_path == '':
            best_model_path = self.get_best_model_path()

        # Try loading the model weights: they must be compatible
        # with the model in memory
        try:
            checkpoint = torch.load(
                best_model_path,
                map_location=torch.device('cpu')
            )
            self._model.load_state_dict(checkpoint)
            print("Loaded best metric model")
        except Exception as e:
            print(type(e))
            self._message = "Error: there was a problem loading the model! Aborting."
            return False

        # If the target folder does not exist, create it
        if prediction_output_path == '':
            prediction_output_path = Path(self._working_dir) / "predictions"
        else:
            prediction_output_path = Path(prediction_output_path)
        prediction_output_path.mkdir(parents=True, exist_ok=True)

        # Switch to evaluation mode
        self._model.eval()

        indx = 0

        # Make sure not to update the gradients
        with torch.no_grad():
            for test_data in self._test_dataloader:

                # Get the next batch and move it to device
                test_images, test_labels = test_data[0].to(self._device), test_data[1].to(self._device)

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

                # Save prediction as npy file
                pred_file_name = os.path.join(
                    str(prediction_output_path),
                    basename + '.npy')
                np.save(pred_file_name, pred)

                # Convert to label image
                label_img = self._prediction_to_label_tiff_image(pred)

                # Save label image as tiff file
                label_file_name = os.path.join(
                    str(prediction_output_path),
                    basename + '.tif')
                with TiffWriter(label_file_name) as tif:
                    tif.save(label_img)

                # Inform
                print(f"Saved {str(prediction_output_path)}/({basename}.npy, {basename}.tif)")

                # Update the index
                indx += 1

        # Inform
        print(f"Prediction completed.")

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

    def _prediction_to_label_tiff_image(self, prediction) -> np.ndarray:
        """Save the prediction to a label image (TIFF)"""
        raise NotImplementedError("Implement me!")

    def _define_transforms(self):
        """Define and initialize data transforms.

        @return True if data transforms could be instantiated, False otherwise.
        """
        raise NotImplementedError("Implement me!")

    def _define_data_loaders(self) -> bool:
        """Initialize datasets and data loaders.

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
        print(f"Using device '{self._device}'.")

        # Try to free memory on the GPU
        if self._device != "cpu":
            torch.cuda.empty_cache()

        # Monai's UNet
        self._model = UNet(
            dimensions=2,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        ).to(self._device)

    def _define_training_loss(self) -> None:
        """Define the loss function."""

        self._training_loss_function = GeneralizedDiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            batch=True,
        )

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
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y%m%d_%H%M%S")

        # Experiment name
        experiment_name = f"{self._raw_experiment_name}_{date_time}" \
            if self._raw_experiment_name != "" \
            else f"{date_time}"
        experiment_name = Path(self._working_dir) / experiment_name

        # Best model file name
        name = Path(self._raw_model_file_name).stem
        model_file_name = f"{name}_{date_time}.pth"
        model_file_name = Path(self._working_dir) / model_file_name

        return str(experiment_name), str(model_file_name)
