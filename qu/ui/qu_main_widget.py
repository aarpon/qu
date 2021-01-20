from pathlib import Path

import torch
from PyQt5 import uic, QtGui
from PyQt5.QtCore import pyqtSlot, QThreadPool, QProcess, QUrl
from PyQt5.QtGui import QIcon, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import QWidget, QFileDialog, QAction, QMessageBox
from torch import __version__ as __torch_version__
from monai import __version__ as __monai_version__
import sys

from qu import __version__
from qu.demo import get_demo_segmentation_dataset
from qu.ml import UNet2DSegmenter
from qu.ml import UNet2DSegmenterSettings
from qu.ui import _ui_folder_path
from qu.console import EmittingErrorStream, EmittingOutputStream
from qu.data import DataModel
from qu.ui.qu_logger_widget import QuLoggerWidget
from qu.ui.qu_unet_settings_dialog import QuUNetSettingsDialog
from qu.ui.threads import LearnerManager, PredictorManager


class QuMainWidget(QWidget):

    def __init__(self, viewer, *args, **kwargs):
        """Constructor."""

        # Call base constructor
        super().__init__(*args, **kwargs)

        # Store a reference to the napari viewer
        self._viewer = viewer

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

        # Create the logger
        self._logger = QuLoggerWidget(viewer)

        # Set up the menu
        self._add_qu_menu()

        # Set the connections
        self._set_connections()

        # Initialize data model
        self._data_model = DataModel()

        # Keep a reference to the learner
        self._learner = None

        # Keep a reference to the settings for the learner (defaults to UNet2DSegmenterSettings)
        self._learner_settings = UNet2DSegmenterSettings()

        # Dock it
        viewer.window.add_dock_widget(self._logger, name='Qu Logger', area='bottom')

        # Tensorboard process
        self._tensorboard_process = None

        # Test redirection to output
        print(f"Welcome to Qu {__version__}.", file=self._out_stream)

    def __del__(self):

        # Restore the streams
        sys.stdout = self._original_out_stream
        sys.stderr = self._original_err_stream

    def _add_qu_menu(self):
        """Add the Qu menu to the main window."""

        # First add a separator from the standard napari menu
        qu_menu = self._viewer.window.main_menu.addMenu(" | ")
        qu_menu.setEnabled(False)

        # Now add the Qu menu
        qu_menu = self._viewer.window.main_menu.addMenu("Qu")

        # About action
        about_action = QAction(QIcon(":/icons/info.png"), "About Qu", self)
        about_action.triggered.connect(self._on_qu_about_action)
        qu_menu.addAction(about_action)

        # Add separator
        qu_menu.addSeparator()

        # Add processing submenu
        processing_menu = qu_menu.addMenu("Processing")

        # Add placeholder for now
        will_follow_action = QAction("Will follow", self)
        processing_menu.addAction(will_follow_action)

        # Add curation submenu
        curation_menu = qu_menu.addMenu("Curation")

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
        qu_menu.addSeparator()

        # Add tools submenu
        tools_menu = qu_menu.addMenu("Tools")

        # Add placeholder for now
        launch_tensorboard_action = QAction("Launch tensorboard", self)
        launch_tensorboard_action.triggered.connect(self._on_launch_tensorboard_action)
        tools_menu.addAction(launch_tensorboard_action)

        # Add separator
        qu_menu.addSeparator()

        # Add demos menu
        demos_menu = qu_menu.addMenu("Demos")

        # Add get demo
        demo_segmentation_action = QAction(QIcon(":/icons/download.png"), "Demo segmentation dataset", self)
        demo_segmentation_action.triggered.connect(self._on_qu_demo_segmentation_action)
        demos_menu.addAction(demo_segmentation_action)

        # Add help action
        help_action = QAction(QIcon(":/icons/help.png"), "Help", self)
        help_action.triggered.connect(self._on_qu_help_action)
        qu_menu.addAction(help_action)

    def _set_connections(self):
        """Connect signals and slots."""

        # Set up connections for UI elements

        # Data root and navigation
        self.pBSelectDataRootFolder.clicked.connect(self._on_select_data_root_folder)
        self.hsImageSelector.valueChanged.connect(self._on_selector_value_changed)

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
        if self._data_model.num_images == 0:
            self.hsImageSelector.setMinimum(0)
            self.hsImageSelector.setMaximum(0)
            self.hsImageSelector.setValue(0)
            self.hsImageSelector.setEnabled(False)
        else:
            self.hsImageSelector.setMinimum(0)
            self.hsImageSelector.setMaximum(self._data_model.num_images - 1)
            self.hsImageSelector.setValue(self._data_model.index)
            self.hsImageSelector.setEnabled(True)

    def display(self) -> None:
        """Display current image and mask."""

        # Get current data (if there is any)
        image, mask = self._data_model.get_image_and_mask_at_current_index()
        if image is None:
            self._update_data_selector()
            return

        # Display image and mask
        if 'Image' in self._viewer.layers:
            self._viewer.layers["Image"].data = image
        else:
            self._viewer.add_image(image, name="Image")

        if 'Mask' in self._viewer.layers:
            self._viewer.layers["Mask"].data = mask
        else:
            self._viewer.add_labels(mask, name="Mask")

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
        training_total_str = f"Training ({training_total}% of {self._data_model.num_images})"
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

    @pyqtSlot(bool, name="_on_select_data_root_folder")
    def _on_select_data_root_folder(self) -> None:
        """Ask the user to pick a data folder."""

        # Check whether we already have data loaded
        if self._data_model.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_model.reset()

        # Ask the user to pick a folder
        output_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Data Root Directory..."
        )
        if output_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataModel
        self._data_model.root_data_path = output_dir

        # Retrieve the (parsed) root data path
        root_data_path = self._data_model.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_model.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_model.preview_training_split()
        self._update_training_ui_elements(
            self._data_model.training_fraction,
            self._data_model.validation_fraction,
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
        predictorManager = PredictorManager(
            self._learner,
            self._data_model.prediction_input_path,
            self._data_model.prediction_target_path,
            self._data_model.model_path
        )

        # Run the training in a separate Qt thread
        predictorManager.signals.started.connect(self._on_prediction_start)
        predictorManager.signals.errored.connect(self._on_prediction_error)
        predictorManager.signals.finished.connect(self._on_prediction_completed)
        predictorManager.signals.returned.connect(self._on_prediction_returned)
        QThreadPool.globalInstance().start(predictorManager)

    @pyqtSlot(name='_on_select_input_for_prediction')
    def _on_select_input_for_prediction(self):
        """Select input folder for prediction."""

        # Ask the user to pick a folder
        input_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Prediction Input Directory..."
        )
        if input_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataModel
        self._data_model.prediction_input_path = input_dir

        # Retrieve the (parsed) input prediction path
        prediction_input_path = self._data_model.prediction_input_path

        # Update the button
        self.pbSelectPredictionDataFolder.setText(str(prediction_input_path))

    @pyqtSlot(name="_on_select_model_for_prediction")
    def _on_select_model_for_prediction(self):
        """Select model for prediction."""

        # Ask the user to pick a model file
        model_file = QFileDialog.getOpenFileName(
            None,
            "Pick a *.pth model file",
            filter="*.pth"
        )

        if model_file[0] == '':
            # The user cancelled the selection
            return

        # Set the path in the DataModel
        self._data_model.model_path = model_file[0]

        # Retrieve the (parsed) model file path
        model_path = self._data_model.model_path

        # Update the button
        self.pbSelectPredictionModelFile.setText(str(model_path))

    @pyqtSlot(name='_on_select_target_for_prediction')
    def _on_select_target_for_prediction(self):
        """Select target folder for prediction."""

        # Ask the user to pick a folder
        target_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Prediction Target Directory..."
        )
        if target_dir == '':
            # The user cancelled the selection
            return

        # Set the path in the DataModel
        self._data_model.prediction_target_path = target_dir

        # Retrieve the (parsed) target prediction path
        prediction_target_path = self._data_model.prediction_target_path

        # Update the button
        self.pbSelectPredictionTargetFolder.setText(str(prediction_target_path))

    @pyqtSlot(str, name='_on_print_output_to_console')
    def _on_print_output_to_console(self, text: str) -> None:
        """Redirect standard output to console."""

        # Append the text
        self._logger.moveCursor(QtGui.QTextCursor.End)
        self._logger.insertPlainText(text)

    @pyqtSlot(str, name='_on_print_error_to_console')
    def _on_print_error_to_console(self, text: str) -> None:
        """Redirect standard error to console."""

        # Get current color
        current_color = self._logger.textColor()

        # Set the color to red
        self._logger.setTextColor(QtGui.QColor(255, 0, 0))

        # Append the text
        self._logger.moveCursor(QtGui.QTextCursor.End)
        self._logger.insertPlainText(text)

        # Restore the color
        self._logger.setTextColor(current_color)

    @pyqtSlot(int, name="_on_selector_value_changed")
    def _on_selector_value_changed(self, value):
        """Triggered when the value of the image selector slider changes.

        Please notice that the slider has tracking off, meaning the
        value_changed signal is not emitted while the slider is being
        moved, but only at the end of the movement!
        """

        # Update the index in the data model
        self._data_model.index = value

        # Update the display
        self.display()

        # Display the image/mask number over the the slider
        self.labelImageCounterCurrent.setText(str(value))

    @pyqtSlot(int, name="_on_train_val_split_selector_value_changed")
    def _on_train_val_split_selector_value_changed(self, value):
        """Recalculate splits and update UI elements."""

        # Update the training fraction in the data model
        self._data_model.training_fraction = float(value) / 100.0

        # Recalculate splits
        num_train, num_val, num_test = self._data_model.preview_training_split()

        # Update UI elements
        self._update_training_ui_elements(
            self._data_model.training_fraction,
            self._data_model.validation_fraction,
            num_train,
            num_val,
            num_test
        )

    @pyqtSlot(int, name="_on_val_test_split_selector_value_changed")
    def _on_val_test_split_selector_value_changed(self, value):
        """Recalculate splits and update UI elements."""

        # Update the validation fraction in the data model
        self._data_model.validation_fraction = float(value) / 100.0

        # Recalculate splits
        num_train, num_val, num_test = self._data_model.preview_training_split()

        # Update UI elements
        self._update_training_ui_elements(
            self._data_model.training_fraction,
            self._data_model.validation_fraction,
            num_train,
            num_val,
            num_test
        )

    @pyqtSlot(name="_on_open_settings_dialog")
    def _on_open_settings_dialog(self):
        """Open settings dialog for currently selected architecture."""

        # Get index of selected architecture
        arch = self.cbArchitecturePicker.currentIndex()

        if arch == 0:
            settings_copy = QuUNetSettingsDialog.get_settings(self._learner_settings)

        if settings_copy is not None:
            self._learner_settings = settings_copy

    @pyqtSlot(name="_on_run_training")
    def _on_run_training(self):
        """Instantiate the Learner (if needed) and run the training."""

        # Is there data to train on?
        if self._data_model.num_images == 0:
            print("Training: please load a dataset first!", file=self._err_stream)
            return

        # Get index of selected architecture
        arch = self.cbArchitecturePicker.currentIndex()

        # Instantiate the requested learner
        if arch == 0:
            self._learner = UNet2DSegmenter(
                self._data_model.mask_type,
                in_channels=self._data_model.num_channels,
                out_channels=self._data_model.num_classes,
                roi_size=self._learner_settings.roi_size,
                num_epochs=self._learner_settings.num_epochs,
                batch_sizes=self._learner_settings.batch_sizes,
                num_workers=self._learner_settings.num_workers,
                validation_step=self._learner_settings.validation_step,
                sliding_window_batch_size=self._learner_settings.sliding_window_batch_size,
                working_dir=self._data_model.root_data_path,
                stdout=self._out_stream,
                stderr=self._err_stream
            )

        # Get the data
        try:
            train_image_names, train_mask_names, \
                val_image_names, val_mask_names, \
                test_image_names, test_mask_names = self._data_model.training_split()
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
        learnerManager = LearnerManager(self._learner)

        # Run the training in a separate Qt thread
        learnerManager.signals.started.connect(self._on_training_start)
        learnerManager.signals.errored.connect(self._on_training_error)
        learnerManager.signals.finished.connect(self._on_training_completed)
        learnerManager.signals.returned.connect(self._on_training_returned)
        QThreadPool.globalInstance().start(learnerManager)

    @pyqtSlot(name="_on_qu_about_action")
    def _on_qu_about_action(self):
        """Qu about action."""
        print(f"Qu version {__version__}", file=self._out_stream)
        print(f"pytorch version {__torch_version__}", file=self._out_stream)
        print(f"monai version {__monai_version__}", file=self._out_stream)

    @pyqtSlot(name="_on_qu_save_mask_action")
    def _on_qu_save_mask_action(self):
        """Qu save mask action."""

        # Save current mask
        if self._data_model.save_mask_at_current_index():
            print(f"Current mask saved.", file=self._out_stream)
        else:
            print(self._data_model.last_error_message, file=self._err_stream)


    @pyqtSlot(name="_on_qu_reload_mask_action")
    def _on_qu_reload_mask_action(self):
        """Qu reload mask action."""

        # Reload mask
        if self._data_model.reload_mask_at_current_index():
            print(f"Current mask reloaded.", file=self._out_stream)
        else:
            print(self._data_model.last_error_message, file=self._err_stream)

        # Update the display
        self.display()

    @pyqtSlot(name="_on_qu_save_all_masks_action")
    def _on_qu_save_all_masks_action(self):
        """Qu save all masks action."""

        # Are there masks loaded?
        if self._data_model.num_masks == 0:
            return

        # Save all cached (i.e. potentially modified) masks
        if self._data_model.save_all_cached_masks():
            print("Saving completed.", file=self._out_stream)
        else:
            print(self._data_model.last_error_message, file=self._err_stream)

    @pyqtSlot(name="_on_launch_tensorboard_action")
    def _on_launch_tensorboard_action(self):
        """Launch tensorboard action."""

        if self._data_model.root_data_path == "":
            print("Select a data folder first.", file=self._err_stream)
            return

        # Check if tensorboard needs to be started
        if self._tensorboard_process is None:
            # It tensorboard is not running, start it and then open the system browser
            self._tensorboard_process = QProcess()
            self._tensorboard_process.readyReadStandardError.connect(self.handle_qprocess_stderr)
            self._tensorboard_process.readyReadStandardOutput.connect(self.handle_qprocess_stdout)
            self._tensorboard_process.start("tensorboard", [f"--logdir={self._data_model.root_data_path}/runs"])
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

    @pyqtSlot(name="_on_qu_demo_segmentation_action")
    def _on_qu_demo_segmentation_action(self):
        """Qu segmentation demo action."""

        # Check whether we already have data loaded
        if self._data_model.num_images > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("Are you sure you want to discard current data?")
            msg.setInformativeText("All data and changes will be lost.")
            msg.setWindowTitle("Qu:: Question")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() == QMessageBox.Cancel:
                return

            # Reset current model
            self._data_model.reset()

        # Inform
        print("If needed, download and extract demo data.", file=self._out_stream)

        # Get the data
        demo_dataset_path = get_demo_segmentation_dataset()

        # Inform
        print("Opening demo dataset.", file=self._out_stream)

        # Set the path in the DataModel
        self._data_model.root_data_path = demo_dataset_path

        # Retrieve the (parsed) root data path
        root_data_path = self._data_model.root_data_path

        # Update the button
        self.pBSelectDataRootFolder.setText(str(root_data_path))

        # Scan the data folder
        try:
            self._data_model.scan()
        except FileNotFoundError as fe:
            print(f"Error: {fe}", file=self._err_stream)
            return
        except ValueError as ve:
            print(f"Error: {ve}", file=self._err_stream)
            return

        # Update the data selector
        self._update_data_selector()

        # Update the training/validation/test split sliders
        num_train, num_val, num_test = self._data_model.preview_training_split()
        self._update_training_ui_elements(
            self._data_model.training_fraction,
            self._data_model.validation_fraction,
            num_train,
            num_val,
            num_test
        )

        # Display current data
        self.display()

        # Inform
        print("Demo dataset opened.", file=self._out_stream)


    @pyqtSlot(name="_on_qu_help_action")
    def _on_qu_help_action(self):
        """Qu help action."""
        print("Help: Implement me!", file=self._out_stream)

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
            # Store the best model path in the data model
            self._data_model.model_path = self._learner.get_best_model_path()

            # Show the name of the best model as text on the "Select model" button and
            # the full path as its tooltip
            self.pbSelectPredictionModelFile.setText(Path(self._data_model.model_path).name)
            self.pbSelectPredictionModelFile.setToolTip(str(self._data_model.model_path))

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
