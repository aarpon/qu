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

import napari
import torch
from PyQt5 import uic, QtGui
from PyQt5.QtCore import pyqtSlot, QThreadPool, QProcess, QUrl, Qt
from PyQt5.QtGui import QIcon, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import QHBoxLayout, QMenu, QPushButton, QWidget, QFileDialog, QAction, QMessageBox, QInputDialog
from torch import __version__ as __torch_version__
from monai import __version__ as __monai_version__
import sys

from qu import __version__
from qu.demo import get_demo_segmentation_dataset, get_demo_restoration_dataset
from qu.models import UNet2DSegmenter, UNet2DRestorer, UNet2DRestorerSettings, UNet2DSegmenterSettings
from qu.processing import SegmentationDiagnostics
from qu.processing.data.images import find_global_intensity_range
from qu.ui import _ui_folder_path
from qu.console import EmittingErrorStream, EmittingOutputStream
from qu.data import DataManager, ExperimentType
from qu.ui.dialogs.qu_unet_restorer_settings_dialog import QuUNetMapperSettingsDialog
from qu.ui.dialogs.qu_unet_segmenter_settings_dialog import QuUNetSegmenterSettingsDialog
from qu.ui.threads import LearnerManager, PredictorManager, SegmentationDiagnosticsManager


class QuMainWidget(QWidget):

    def __init__(self, *args, **kwargs):
        """Constructor."""

        # Call base constructor
        super().__init__(*args, **kwargs)

        # Get current napari viewer
        self._viewer = napari.current_viewer()

        # Initialize data manager
        self._data_manager = DataManager()

        # Keep references to all learners and their settings
        self._all_learners_settings = {
            0: (UNet2DSegmenter(), UNet2DSegmenterSettings()),
            1: (UNet2DRestorer(), UNet2DRestorerSettings())
        }

        # Keep reference to the learner (instantiate to the first one)
        self._learner = self._all_learners_settings[0][0]

        # Keep a reference to the settings for the learner (instantiate to the first one)
        self._learner_settings = self._all_learners_settings[0][1]

        # Keep a reference to the segmentation diagnostics tool
        self._segm_diagnostics = None

        # Set up UI
        uic.loadUi(_ui_folder_path / "qu_main_widget.ui", self)

        # Keep a reference to the stream objects
        self._original_out_stream = sys.stdout
        self._original_err_stream = sys.stderr
        self._out_stream = EmittingOutputStream()
        self._err_stream = EmittingErrorStream()

        # Setup standard out and err redirection
        sys.stdout = self._out_stream
        sys.stdout.stream_signal.connect(self._on_print_output_to_console)
        sys.stderr = self._err_stream
        sys.stderr.stream_signal.connect(self._on_print_error_to_console)

        # Make sure that the logger is read only
        self.teLogger.setReadOnly(True)
        self.teLogger.setAcceptDrops(False)

        # Set up the context menu
        self._qu_context_menu = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        # Set the connections
        self._set_connections()

        # Tensorboard process
        self._tensorboard_process = None

        # Test redirection to output
        if torch.cuda.is_available():
            print(f"Welcome to Qu {__version__} using {torch.cuda.get_device_name()}.", file=self._out_stream)
        else:
            print(f"Welcome to Qu {__version__} (no GPU found).", file=self._out_stream)
        
        # Inform the user about the context menu
        print(f"\nRight-click on the widget for main menu.\n", file=self._out_stream)

    def __del__(self):

        # Restore the streams
        sys.stdout = self._original_out_stream
        sys.stderr = self._original_err_stream

    def _show_qu_context_menu(self, pos):
        """Add the Qu context menu to the widget."""

        # Create menu if not yet ready
        if self._qu_context_menu is None:

            self._qu_context_menu = QMenu()

            # About action
            about_action = QAction(QIcon(":/icons/info.png"), "About Qu", self)
            about_action.triggered.connect(self._on_qu_about_action)
            self._qu_context_menu.addAction(about_action)

            # Add separator
            self._qu_context_menu.addSeparator()

            # Add processing submenu
            processing_menu = self._qu_context_menu.addMenu("Processing")

            # Add images submenu
            images_menu = processing_menu.addMenu("Images")

            # Save mask
            find_global_intensity_range_action = QAction("Find global intensity range", self)
            find_global_intensity_range_action.triggered.connect(self._on_find_global_intensity_range_action)
            images_menu.addAction(find_global_intensity_range_action)

            # Add masks submenu
            masks_menu = processing_menu.addMenu("Masks")

            # Add placeholders for now
            will_follow_action = QAction("Will follow", self)
            masks_menu.addAction(will_follow_action)

            # Add curation submenu
            curation_menu = self._qu_context_menu.addMenu("Curation")

            # Save mask
            save_mask_action = QAction(QIcon(":/icons/save.png"), "Save mask", self)
            save_mask_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
            save_mask_action.triggered.connect(self._on_qu_save_mask_action)
            curation_menu.addAction(save_mask_action)

            # Reload mask
            reload_mask_action = QAction(QIcon(":/icons/revert.png"), "Reload mask", self)
            reload_mask_action.setShortcut(QKeySequence("Ctrl+Alt+Z"))
            reload_mask_action.triggered.connect(self._on_qu_reload_mask_action)
            curation_menu.addAction(reload_mask_action)

            # Add separator
            curation_menu.addSeparator()

            # Save mask
            save_all_mask_action = QAction(QIcon(":/icons/save.png"), "Save all masks", self)
            save_all_mask_action.triggered.connect(self._on_qu_save_all_masks_action)
            curation_menu.addAction(save_all_mask_action)

            # Add separator
            self._qu_context_menu.addSeparator()

            # Add Diagnostics submenu
            diagnostics_menu = self._qu_context_menu.addMenu("Diagnostics")

            # Add Launch Tensorboard action
            launch_tensorboard_action = QAction("Launch tensorboard", self)
            launch_tensorboard_action.triggered.connect(self._on_launch_tensorboard_action)
            diagnostics_menu.addAction(launch_tensorboard_action)

            # Add segmentation diagnostics
            seg_diagnostic = QAction("Segmentation diagnostics", self)
            seg_diagnostic.triggered.connect(self._on_qu_segmentation_diagnostic)
            diagnostics_menu.addAction(seg_diagnostic)

            # Add separator
            self._qu_context_menu.addSeparator()

            # Add demos menu
            demos_menu = self._qu_context_menu.addMenu("Demos")

            # Add segmentation demos menu
            segmentation_demos_menu = demos_menu.addMenu("Segmentation dataset")

            # Add demos
            demo_3_classes_segmentation_action = QAction(QIcon(":/icons/download.png"), "3 classes", self)
            demo_3_classes_segmentation_action.triggered.connect(self._on_qu_demo_3_classes_segmentation_action)
            segmentation_demos_menu.addAction(demo_3_classes_segmentation_action)

            demo_2_classes_segmentation_action = QAction(QIcon(":/icons/download.png"), "2 classes", self)
            demo_2_classes_segmentation_action.triggered.connect(self._on_qu_demo_2_classes_segmentation_action)
            segmentation_demos_menu.addAction(demo_2_classes_segmentation_action)

            # demos_menu.addMenu(segmentation_demos_menu)

            demo_restoration_action = QAction(QIcon(":/icons/download.png"), "Restoration dataset", self)
            demo_restoration_action.triggered.connect(self._on_qu_demo_restoration_action)
            demos_menu.addAction(demo_restoration_action)

            # Add help action
            help_action = QAction(QIcon(":/icons/help.png"), "Help", self)
            help_action.triggered.connect(self._on_qu_help_action)
            self._qu_context_menu.addAction(help_action)

        # Execute the action
        self._qu_context_menu.exec_(self.mapToGlobal(pos))

    def _set_connections(self):
        """Connect signals and slots."""

        # Set up connections for UI elements

        # Context menu
        self.customContextMenuRequested.connect(self._show_qu_context_menu)

        # Data root and navigation
        self.pBSelectDataRootFolder.clicked.connect(self._on_select_data_root_folder)
        self.hsImageSelector.valueChanged.connect(self._on_selector_value_changed)

        # Architecture picker
        self.cbArchitecturePicker.currentIndexChanged.connect(self._on_architecture_changed)

        # Training
        self.hsTrainingValidationSplit.valueChanged.connect(self._on_train_val_split_selector_value_changed)
        self.hsValidationTestingSplit.valueChanged.connect(self._on_val_test_split_selector_value_changed)
        self.pbTrain.clicked.connect(self._on_run_training)
        self.pbArchitectureSettings.clicked.connect(self._on_open_settings_dialog)

        # Prediction
        self.pbSelectPredictionDataFolder.clicked.connect(self._on_select_input_for_prediction)
        self.pbSelectPredictionTargetFolder.clicked.connect(self._on_select_target_for_prediction)
        self.pbSelectPredictionModelFile.clicked.connect(self._on_select_model_for_prediction)
        self.pbApplyPrediction.clicked.connect(self._on_run_prediction)

        # Other
        self.pbFreeMemory.clicked.connect(self._on_free_memory_and_report)

    def _update_data_selector(self) -> None:
        """Update data selector (slider)."""

        # Update the slider
        if self._data_manager.num_images == 0:
            self.hsImageSelector.setMinimum(0)
            self.hsImageSelector.setMaximum(0)
            self.hsImageSelector.setValue(0)
            self.hsImageSelector.setEnabled(False)
        else:
            self.hsImageSelector.setMinimum(0)
            self.hsImageSelector.setMaximum(self._data_manager.num_images - 1)
            self.hsImageSelector.setValue(self._data_manager.index)
            self.hsImageSelector.setEnabled(True)

    def display(self) -> None:
        """Display current image and mask."""

        # Get current data (if there is any)
        image, out = self._data_manager.get_image_data_at_current_index()
        if image is None:
            self._update_data_selector()
            return

        # Display image and mask/target
        if 'Image' in self._viewer.layers:
            self._viewer.layers["Image"].data = image
        else:
            self._viewer.add_image(image, name="Image")

        if self._data_manager.experiment_type == ExperimentType.CLASSIFICATION:
            if 'Mask' in self._viewer.layers:
                self._viewer.layers["Mask"].data = out
            else:
                self._viewer.add_labels(out, name="Mask")
        elif self._data_manager.experiment_type == ExperimentType.REGRESSION:
            if 'Target' in self._viewer.layers:
                self._viewer.layers["Target"].data = out
            else:
                self._viewer.add_image(out, name="Target")
        else:
            raise Exception("Unknown experiment type.")

    def _update_training_ui_elements(
            self,
            training_fraction,
            validation_fraction,
            num_train,
            num_val,
            num_test
    ):
        """Updates the ui elements associated with the training parameters."""

        training_total = int(100 * training_fraction)
        validation_total = int(100 * validation_fraction)
        training_total_str = f"Training ({training_total}% of {self._data_manager.num_images})"
        validation_total_str = f"Validation:Test ({validation_total}%)"
        num_training_images = f"{num_train} training images."
        num_val_test_images = f"{num_val}:{num_test} val:test images."

        # Update the training elements
        self.lbTrainingValidationSplit.setText(training_total_str)
        self.hsTrainingValidationSplit.blockSignals(True)
        self.hsTrainingValidationSplit.setValue(training_total)
        self.hsTrainingValidationSplit.blockSignals(False)
        self.hsTrainingValidationSplit.setEnabled(True)
        self.lbNumberTrainingImages.setText(num_training_images)

        # Update the validation/test elements
        self.lbValidationTestingSplit.setText(validation_total_str)
        self.hsValidationTestingSplit.blockSignals(True)
        self.hsValidationTestingSplit.setValue(validation_total)
        self.hsValidationTestingSplit.blockSignals(False)
        self.hsValidationTestingSplit.setEnabled(True)
        self.lbNumberValidationTestingImages.setText(num_val_test_images)

    def _setup_session(self, data_folder):
        """Set up a new session using the specified data folder."""

    @pyqtSlot(int, name="_on_architecture_changed")
    def _on_architecture_changed(self, new_index) -> None:
        """Called then the selection in the architecture pull-down menu changes."""

        # Get index of selected architecture
        arch = self.cbArchitecturePicker.currentIndex()

        # Update the references
        self._learner = self._all_learners_settings[arch][0]
        self._learner_settings = self._all_learners_settings[arch][1]

    @pyqtSlot(bool, name="_on_select_data_root_folder")
    def _on_select_data_root_folder(self) -> None:
        """Ask the user to pick a data folder."""

        # Check whether we already have data loaded
        if self._data_manager.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_manager.reset()

        # Ask the user to pick a folder
        output_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Data Root Directory..."
        )
        if output_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataManager
        self._data_manager.root_data_path = output_dir

        # Retrieve the (parsed) root data path
        root_data_path = self._data_manager.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_manager.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_manager.preview_training_split()
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

        # Display current data
        self.display()

    @pyqtSlot(name='_on_run_prediction')
    def _on_run_prediction(self):
        """Run prediction."""

        # Instantiate the manager
        predictor_manager = PredictorManager(
            self._learner,
            self._data_manager.prediction_input_path,
            self._data_manager.prediction_target_path,
            self._data_manager.model_path
        )

        # Run the training in a separate Qt thread
        predictor_manager.signals.started.connect(self._on_prediction_start)
        predictor_manager.signals.errored.connect(self._on_prediction_error)
        predictor_manager.signals.finished.connect(self._on_prediction_completed)
        predictor_manager.signals.returned.connect(self._on_prediction_returned)
        QThreadPool.globalInstance().start(predictor_manager)

    @pyqtSlot(name='_on_select_input_for_prediction')
    def _on_select_input_for_prediction(self):
        """Select input folder for prediction."""

        default_folder = str(
            self._data_manager.root_data_path if
            self._data_manager.root_data_path is not None else Path(".")
        )

        # Ask the user to pick a folder
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Prediction Input Directory...",
            default_folder,
            QFileDialog.ShowDirsOnly
        )
        if input_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataManager
        self._data_manager.prediction_input_path = input_dir

        # Retrieve the (parsed) input prediction path
        prediction_input_path = self._data_manager.prediction_input_path

        # Update the button
        self.pbSelectPredictionDataFolder.setText(str(prediction_input_path))

    @pyqtSlot(name="_on_select_model_for_prediction")
    def _on_select_model_for_prediction(self):
        """Select model for prediction."""

        default_folder = str(
            self._data_manager.root_data_path / "runs" if
            self._data_manager.root_data_path is not None else Path(".")
        )

        # Ask the user to pick a model file
        model_file = QFileDialog.getOpenFileName(
            self,
            "Pick a *.pth model file",
            default_folder,
            filter="*.pth"
        )

        if model_file[0] == '':
            # The user cancelled the selection
            return

        # Set the path in the DataManager
        self._data_manager.model_path = model_file[0]

        # Retrieve the (parsed) model file path
        model_path = self._data_manager.model_path

        # Update the button
        self.pbSelectPredictionModelFile.setText(str(model_path.name))

    @pyqtSlot(name='_on_select_target_for_prediction')
    def _on_select_target_for_prediction(self):
        """Select target folder for prediction."""

        default_folder = str(
            self._data_manager.root_data_path if
            self._data_manager.root_data_path is not None else Path(".")
        )

        # Ask the user to pick a folder
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Prediction Target Directory...",
            default_folder,
            QFileDialog.ShowDirsOnly
        )
        if target_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataManager
        self._data_manager.prediction_target_path = target_dir

        # Retrieve the (parsed) target prediction path
        prediction_target_path = self._data_manager.prediction_target_path

        # Update the button
        self.pbSelectPredictionTargetFolder.setText(str(prediction_target_path))

    @pyqtSlot(str, name='_on_print_output_to_console')
    def _on_print_output_to_console(self, text: str) -> None:
        """Redirect standard output to console."""

        # Append the text
        self.teLogger.moveCursor(QtGui.QTextCursor.End)
        self.teLogger.insertPlainText(text)

    @pyqtSlot(str, name='_on_print_error_to_console')
    def _on_print_error_to_console(self, text: str) -> None:
        """Redirect standard error to console."""

        # Get current color
        current_color = self.teLogger.textColor()

        # Set the color to red
        self.teLogger.setTextColor(QtGui.QColor(255, 0, 0))

        # Append the text
        self.teLogger.moveCursor(QtGui.QTextCursor.End)
        self.teLogger.insertPlainText(text)

        # Restore the color
        self.teLogger.setTextColor(current_color)

    @pyqtSlot(int, name="_on_selector_value_changed")
    def _on_selector_value_changed(self, value):
        """Triggered when the value of the image selector slider changes.

        Please notice that the slider has tracking off, meaning the
        value_changed signal is not emitted while the slider is being
        moved, but only at the end of the movement!
        """

        # Update the index in the data manager
        self._data_manager.index = value

        # Update the display
        self.display()

        # Display the image/mask number over the the slider
        self.labelImageCounterCurrent.setText(str(value))

    @pyqtSlot(int, name="_on_train_val_split_selector_value_changed")
    def _on_train_val_split_selector_value_changed(self, value):
        """Recalculate splits and update UI elements."""

        # Update the training fraction in the data manager
        self._data_manager.training_fraction = float(value) / 100.0

        # Recalculate splits
        num_train, num_val, num_test = self._data_manager.preview_training_split()

        # Update UI elements
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

    @pyqtSlot(int, name="_on_val_test_split_selector_value_changed")
    def _on_val_test_split_selector_value_changed(self, value):
        """Recalculate splits and update UI elements."""

        # Update the validation fraction in the data manager
        self._data_manager.validation_fraction = float(value) / 100.0

        # Recalculate splits
        num_train, num_val, num_test = self._data_manager.preview_training_split()

        # Update UI elements
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

    @pyqtSlot(name="_on_open_settings_dialog")
    def _on_open_settings_dialog(self):
        """Open settings dialog for currently selected architecture."""

        # Get index of selected architecture
        arch = self.cbArchitecturePicker.currentIndex()

        # Open the dialog and allow the user to modify the settings.
        # When the dialog is closed, the updated settings are returned.
        if arch == 0:
            settings_copy = QuUNetSegmenterSettingsDialog.get_settings(self._all_learners_settings[arch][1])
        elif arch == 1:
            settings_copy = QuUNetMapperSettingsDialog.get_settings(self._all_learners_settings[arch][1])
        else:
            raise Exception("Unsupported option!")

        # If the user did not "cancel" the dialog, update the settings references
        if settings_copy is not None:
            self._learner_settings = settings_copy
            new_settings_tuple = (self._all_learners_settings[arch][0], settings_copy)
            self._all_learners_settings[arch] = new_settings_tuple

    @pyqtSlot(name="_on_run_training")
    def _on_run_training(self):
        """Instantiate the Learner (if needed) and run the training."""

        # Is there data to train on?
        if self._data_manager.num_images == 0:
            print("Training: please load a dataset first!", file=self._err_stream)
            return

        # Get index of selected architecture
        arch = self.cbArchitecturePicker.currentIndex()

        # Instantiate the requested learner
        if arch == 0:
            self._learner = UNet2DSegmenter(
                architecture=self._learner_settings.architecture,
                loss=self._learner_settings.loss,
                optimizer=self._learner_settings.optimizer,
                mask_type=self._data_manager.mask_type,
                in_channels=self._data_manager.num_input_channels,
                out_channels=self._data_manager.num_classes,
                roi_size=self._learner_settings.roi_size,
                learning_rate=self._learner_settings.learning_rate,
                weight_decay=self._learner_settings.weight_decay,
                momentum=self._learner_settings.momentum,
                num_epochs=self._learner_settings.num_epochs,
                batch_sizes=self._learner_settings.batch_sizes,
                num_workers=self._learner_settings.num_workers,
                validation_step=self._learner_settings.validation_step,
                sliding_window_batch_size=self._learner_settings.sliding_window_batch_size,
                working_dir=self._data_manager.root_data_path,
                stdout=self._out_stream,
                stderr=self._err_stream
            )
        elif arch == 1:
            self._learner = UNet2DRestorer(
                architecture=self._learner_settings.architecture,
                loss=self._learner_settings.loss,
                optimizer=self._learner_settings.optimizer,
                in_channels=self._data_manager.num_input_channels,
                out_channels=self._data_manager.num_output_channels,
                norm_min=self._learner_settings.norm_min,
                norm_max=self._learner_settings.norm_max,
                num_samples=self._learner_settings.num_samples,
                roi_size=self._learner_settings.roi_size,
                learning_rate=self._learner_settings.learning_rate,
                weight_decay=self._learner_settings.weight_decay,
                momentum=self._learner_settings.momentum,
                num_epochs=self._learner_settings.num_epochs,
                batch_sizes=self._learner_settings.batch_sizes,
                num_workers=self._learner_settings.num_workers,
                validation_step=self._learner_settings.validation_step,
                sliding_window_batch_size=self._learner_settings.sliding_window_batch_size,
                working_dir=self._data_manager.root_data_path,
                stdout=self._out_stream,
                stderr=self._err_stream
            )
        else:
            raise Exception("Unsupported option!")

        # Get the data
        try:
            train_image_names, train_mask_names, \
            val_image_names, val_mask_names, \
            test_image_names, test_mask_names = self._data_manager.training_split()
        except ValueError as e:
            print(f"Error: {str(e)}. Aborting...", file=self._err_stream)
            return
        except Exception as x:
            print(f"{str(x)}", file=self._err_stream)
            return

        # Pass the training data to the learner
        self._learner.set_training_data(
            train_image_names,
            train_mask_names,
            val_image_names,
            val_mask_names,
            test_image_names,
            test_mask_names
        )

        # Instantiate the manager
        learner_manager = LearnerManager(self._learner)

        # Run the training in a separate Qt thread
        learner_manager.signals.started.connect(self._on_training_start)
        learner_manager.signals.errored.connect(self._on_training_error)
        learner_manager.signals.finished.connect(self._on_training_completed)
        learner_manager.signals.returned.connect(self._on_training_returned)
        QThreadPool.globalInstance().start(learner_manager)

    @pyqtSlot(name="_on_qu_about_action")
    def _on_qu_about_action(self):
        """Qu about action."""
        print(f"Qu version {__version__}", file=self._out_stream)
        print(f"pytorch version {__torch_version__}", file=self._out_stream)
        print(f"monai version {__monai_version__}", file=self._out_stream)

    @pyqtSlot(name="_on_qu_segmentation_diagnostic")
    def _on_qu_segmentation_diagnostic(self):
        """Qu call segmentation diagnostic"""
        # Ask the user to pick the ground truth folder
        gt_dir = QFileDialog.getExistingDirectory(
            None,
            "Select masks (ground truth) directory..."
        )
        if gt_dir == '':
            # The user cancelled the selection
            return

        # Ask the user to pick the segmentation folder
        segm_dir = QFileDialog.getExistingDirectory(
            None,
            "Select prediction directory..."
        )
        if segm_dir == '':
            # The user cancelled the selection
            return

        # Initialize SegmentationDiagnostics object
        self._segm_diagnostics = SegmentationDiagnostics(gt_dir, segm_dir)

        # Wrap the SegmentationDiagnostics tool into its Manager
        segm_manager = SegmentationDiagnosticsManager(self._segm_diagnostics)

        # Run the training in a separate Qt thread
        segm_manager.signals.started.connect(self._on_segm_diagnostics_start)
        segm_manager.signals.errored.connect(self._on_segm_diagnostics_error)
        segm_manager.signals.finished.connect(self._on_segm_diagnostics_completed)
        segm_manager.signals.returned.connect(self._on_segm_diagnostics_returned)
        QThreadPool.globalInstance().start(segm_manager)

    @pyqtSlot(name="_on_find_global_intensity_range_action")
    def _on_find_global_intensity_range_action(self):

        # Check if we have images and (optionally) targets
        if self._data_manager.num_images == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Please load a dataset first.")
            msg.setInformativeText("No images found.")
            msg.setWindowTitle("Qu:: Error")
            msg.setStandardButtons(QMessageBox.Ok)
            if msg.exec_() == QMessageBox.Ok:
                return

        # Ask for the percentile
        perc, ok = QInputDialog.getInt(
            self,
            "Input requested",
            "Intensity percentile (0 - 49):",
            1,
            0,
            49,
            1,
            Qt.Popup
        )

        if not ok:
            return

        # Get the intensitiy range
        mn, mx = find_global_intensity_range(
            Path(self._data_manager.root_data_path) / "images",
            Path(self._data_manager.root_data_path) / "targets",
            perc
        )

        # Inform
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Global intensity range found:")
        msg.setInformativeText(f"\nmin = {mn}\nmax = {mx}")
        msg.setWindowTitle("Qu:: Info")
        msg.setStandardButtons(QMessageBox.Ok)
        if msg.exec_() == QMessageBox.Ok:
            return

    @pyqtSlot(name="_on_qu_save_mask_action")
    def _on_qu_save_mask_action(self):
        """Qu save mask action."""

        # This applies only to classification experiments
        if self._data_manager.experiment_type == ExperimentType.CLASSIFICATION:

            # Save current mask
            if self._data_manager.save_mask_at_current_index():
                print(f"Current mask saved.", file=self._out_stream)
            else:
                print(self._data_manager.last_error_message, file=self._err_stream)

        else:
            print("There are no masks to save.", file=self._err_stream)

    @pyqtSlot(name="_on_qu_reload_mask_action")
    def _on_qu_reload_mask_action(self):
        """Qu reload mask action."""

        # This applies only to classification experiments
        if self._data_manager.experiment_type == ExperimentType.CLASSIFICATION:

            # Reload mask
            if self._data_manager.reload_mask_at_current_index():
                print(f"Current mask reloaded.", file=self._out_stream)
            else:
                print(self._data_manager.last_error_message, file=self._err_stream)

            # Update the display
            self.display()

        else:
            print("There are no masks to reload.", file=self._err_stream)

    @pyqtSlot(name="_on_qu_save_all_masks_action")
    def _on_qu_save_all_masks_action(self):
        """Qu save all masks action."""

        # This applies only to classification experiments
        if self._data_manager.experiment_type == ExperimentType.CLASSIFICATION:

            # Are there masks loaded?
            if self._data_manager.num_masks == 0:
                return

            # Save all cached (i.e. potentially modified) masks
            if self._data_manager.save_all_cached_masks():
                print("Saving completed.", file=self._out_stream)
            else:
                print(self._data_manager.last_error_message, file=self._err_stream)

        else:
            print("There are no masks to save.", file=self._err_stream)

    @pyqtSlot(name="_on_launch_tensorboard_action")
    def _on_launch_tensorboard_action(self):
        """Launch tensorboard action."""

        if self._data_manager.root_data_path == "":
            print("Select a data folder first.", file=self._err_stream)
            return

        # Check if tensorboard needs to be started
        if self._tensorboard_process is None:
            # It tensorboard is not running, start it and then open the system browser
            self._tensorboard_process = QProcess()
            self._tensorboard_process.readyReadStandardError.connect(self.handle_qprocess_stderr)
            self._tensorboard_process.readyReadStandardOutput.connect(self.handle_qprocess_stdout)
            self._tensorboard_process.start("tensorboard", [f"--logdir={self._data_manager.root_data_path}"])
            self._tensorboard_process.started.connect(self._on_open_tensorboard_in_browser)
        else:
            # Tensorboard is already running, just open the browser
            self._on_open_tensorboard_in_browser()

    @pyqtSlot(name="handle_qprocess_stderr")
    def handle_qprocess_stderr(self):
        if self._tensorboard_process is None:
            return
        data = self._tensorboard_process.readAllStandardError()
        message = bytes(data).decode("utf8")
        print(message, file=self._err_stream)

    @pyqtSlot(name="handle_qprocess_stdout")
    def handle_qprocess_stdout(self):
        if self._tensorboard_process is None:
            return
        data = self._tensorboard_process.readAllStandardOutput()
        message = bytes(data).decode("utf8")
        print(message, file=self._out_stream)

    @pyqtSlot(name="_on_open_tensorboard_in_browser")
    def _on_open_tensorboard_in_browser(self):
        """Open tensorboard in browser action."""
        url = QUrl("http://localhost:6006/")
        QDesktopServices.openUrl(url)

    @pyqtSlot(name="_on_qu_demo_3_classes_segmentation_action")
    def _on_qu_demo_3_classes_segmentation_action(self):
        """Qu 3-classes segmentation demo action."""

        # Check whether we already have data loaded
        if self._data_manager.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_manager.reset()

        # Inform
        print("If needed, download and extract segmentation demo data.", file=self._out_stream)

        # Get the data
        demo_dataset_path = get_demo_segmentation_dataset(three_classes=True)

        # Inform
        print("Opening segmentation demo dataset.", file=self._out_stream)

        # Set the path in the DataManager
        self._data_manager.root_data_path = demo_dataset_path

        # Retrieve the (parsed) root data path
        root_data_path = self._data_manager.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_manager.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_manager.preview_training_split()
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

        # Display current data
        self.display()

        # Inform
        print("Segmentation demo dataset opened.", file=self._out_stream)

    @pyqtSlot(name="_on_qu_demo_2_classes_segmentation_action")
    def _on_qu_demo_2_classes_segmentation_action(self):
        """Qu 2-classes segmentation demo action."""

        # Check whether we already have data loaded
        if self._data_manager.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_manager.reset()

        # Inform
        print("If needed, download and extract segmentation demo data.", file=self._out_stream)

        # Get the data
        demo_dataset_path = get_demo_segmentation_dataset(three_classes=False)

        # Inform
        print("Opening segmentation demo dataset.", file=self._out_stream)

        # Set the path in the DataManager
        self._data_manager.root_data_path = demo_dataset_path

        # Retrieve the (parsed) root data path
        root_data_path = self._data_manager.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_manager.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_manager.preview_training_split()
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

        # Display current data
        self.display()

        # Inform
        print("Segmentation demo dataset opened.", file=self._out_stream)

    @pyqtSlot(name="_on_qu_demo_restoration_action")
    def _on_qu_demo_restoration_action(self):
        """Qu restoration demo action."""

        # Check whether we already have data loaded
        if self._data_manager.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_manager.reset()

        # Inform
        print("If needed, download and extract restoration demo data.", file=self._out_stream)

        # Get the data
        demo_dataset_path = get_demo_restoration_dataset()

        # Inform
        print("Opening restoration demo dataset.", file=self._out_stream)

        # Set the path in the DataManager
        self._data_manager.root_data_path = demo_dataset_path

        # Retrieve the (parsed) root data path
        root_data_path = self._data_manager.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_manager.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_manager.preview_training_split()
        self._update_training_ui_elements(
            self._data_manager.training_fraction,
            self._data_manager.validation_fraction,
            num_train,
            num_val,
            num_test
        )

        # Display current data
        self.display()

        # Inform
        print("Restoration demo dataset opened.", file=self._out_stream)

    @pyqtSlot(name="_on_qu_help_action")
    def _on_qu_help_action(self):
        """Qu help action."""
        QDesktopServices.openUrl(QUrl("https://github.com/aarpon/qu/wiki/Qu", QUrl.TolerantMode))

    @pyqtSlot(name="_on_training_start")
    def _on_training_start(self):
        """Called when training is started."""
        print("Training started.", file=self._out_stream)

    @pyqtSlot(name="_on_training_completed")
    def _on_training_completed(self):
        """Called when training is complete."""
        print("All training threads returned.", file=self._out_stream)

    @pyqtSlot(object, name="_on_training_returned")
    def _on_training_returned(self, value):
        """Called when training returned."""
        if bool(value):
            # Store the best model path in the data manager
            self._data_manager.model_path = self._learner.get_best_model_path()

            # Show the name of the best model as text on the "Select model" button and
            # the full path as its tooltip
            self.pbSelectPredictionModelFile.setText(Path(self._data_manager.model_path).name)
            self.pbSelectPredictionModelFile.setToolTip(str(self._data_manager.model_path))

            # Inform
            print(f"Training was successful.", file=self._out_stream)

        else:
            # Inform
            print(f"Training was not successful.", file=self._out_stream)

    @pyqtSlot(object, name="_on_training_error")
    def _on_training_error(self, err):
        """Called if training failed."""
        print(f"Training error: {str(err)}", file=self._out_stream)

    @pyqtSlot(name="_on_prediction_start")
    def _on_prediction_start(self):
        """Called when prediction is started."""
        print("Prediction started.", file=self._out_stream)

    @pyqtSlot(name="_on_prediction_completed")
    def _on_prediction_completed(self):
        """Called when prediction is complete."""
        print("All prediction threads returned.", file=self._out_stream)

    @pyqtSlot(object, name="_on_prediction_returned")
    def _on_prediction_returned(self, value):
        """Called when prediction returned."""
        if bool(value):
            # Inform
            print(f"Prediction was successful.", file=self._out_stream)
        else:
            # Inform
            print(f"Prediction was not successful.", file=self._out_stream)

    @pyqtSlot(object, name="_on_prediction_error")
    def _on_prediction_error(self, err):
        """Called if prediction failed."""
        print(f"Prediction error: {str(err)}", file=self._err_stream)

    @pyqtSlot(name="_on_segm_diagnostics_start")
    def _on_segm_diagnostics_start(self):
        """Called when segmentation diagnostics is started."""
        print("Segmentation diagnostics started. When completed, a figure will be displayed. This may take a while...", file=self._out_stream)

    @pyqtSlot(name="_on_segm_diagnostics_completed")
    def _on_segm_diagnostics_completed(self):
        """Called when segmentation diagnostics is complete."""
        print("All segmentation diagnostics threads returned.", file=self._out_stream)

    @pyqtSlot(object, name="_on_segm_diagnostics_returned")
    def _on_segm_diagnostics_returned(self, value):
        """Called when segmentation diagnostics returned."""
        if value:
            value = str(value)
            # Inform
            print(f"Segmentation diagnostics was successful.", file=self._out_stream)

    @pyqtSlot(object, name="_on_segm_diagnostics_error")
    def _on_segm_diagnostics_error(self, err):
        """Called if segmentation diagnostics failed."""
        print(f"Segmentation diagnostics error: {str(err)}", file=self._err_stream)

    @pyqtSlot(name="_on_free_memory_and_report")
    def _on_free_memory_and_report(self):
        """Try freeing memory on GPU and report."""
        gb = 1024 * 1024 * 1024
        if torch.cuda.is_available():
            n = torch.cuda.get_device_name()
            p = torch.cuda.get_device_capability()
            t = round(torch.cuda.get_device_properties(0).total_memory / gb, 2)
            c = round(torch.cuda.memory_reserved(0) / gb, 2)
            a = round(torch.cuda.memory_allocated(0) / gb, 2)
            torch.cuda.empty_cache()
            c2 = round(torch.cuda.memory_reserved(0) / gb, 2)
            a2 = round(torch.cuda.memory_allocated(0) / gb, 2)
            print(f"{n} (CP {p[0]}.{p[1]}) - "
                  f"Total memory = {t} GB, "
                  f"allocated = {a2} GB (before = {a} GB), "
                  f"reserved = {c2} GB (before = {c} GB)", file=self._out_stream)
        else:
            print("GPU not available.", file=self._out_stream)
