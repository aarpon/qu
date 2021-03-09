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
from typing import Union

from PyQt5.QtCore import pyqtSlot
from napari.qt.threading import WorkerBase


class LearnerManager(WorkerBase):
    """Runs a Learner in a separate Qt thread.

    Extends napari's WorkerBase.
    """

    def __init__(self, learner):
        """Constructor.

        learner: obj
            Learner to be injected.
        """

        # Call base constructor
        super().__init__()

        # Learner
        self._learner = learner

        # Error message
        self._message = ""

    @pyqtSlot()
    def run(self):
        """Run method to be executed in a separate thread."""

        # Emit started signal
        self.signals.started.emit()

        # Global success

        # Run the training
        success = self._learner.train()
        if success:
            # Run the test prediction
            success = self._learner.test_predict()

        if not success:
            # Emit the errored signal with the error message
            self.signals.errored.emit(self._learner.get_message())

        # Emit the finished signal
        self.signals.finished.emit()

        # Emit the returned signal with value `result`
        self.signals.returned.emit(success)


class PredictorManager(WorkerBase):
    """Runs a Predictor in a separate Qt thread.

    Extends napari's WorkerBase.
    """

    def __init__(self,
                 trained_learner,
                 input_folder: Union[Path, str],
                 target_folder: Union[Path, str],
                 model_path: Union[Path, str]
                 ):
        """Constructor.

        @param trained_learner: obj
            Learner to be injected. It assumed to have been already trained.
        """

        # Call base constructor
        super().__init__()

        # Trained learner
        self._trained_learner = trained_learner

        # Relevant paths
        self._input_folder = input_folder
        self._target_folder = target_folder
        self._model_path = model_path

        # Error message
        self._message = ""

    @pyqtSlot()
    def run(self):
        """Run method to be executed in a separate thread."""

        # Emit started signal
        self.signals.started.emit()

        if self._trained_learner is None:
            # Emit the errored signal with the error message
            self.signals.errored.emit("Please train a network first!")

            success = False

        elif self._input_folder == '':
            # Emit the errored signal with the error message
            self.signals.errored.emit("Please specify an input folder for prediction!")

            success = False

        elif self._target_folder == '':
            # Emit the errored signal with the error message
            self.signals.errored.emit("Please specify a target folder for prediction!")

            success = False

        elif self._model_path == '':
            # Emit the errored signal with the error message
            self.signals.errored.emit("Please specify a model file to be used for prediction!")

            success = False

        else:

            # Run the prediction
            success = self._trained_learner.predict(
                self._input_folder,
                self._target_folder,
                self._model_path
            )

            if not success:
                # Emit the errored signal with the error message
                self.signals.errored.emit(self._learner.get_message())

        # Emit the finished signal
        self.signals.finished.emit()

        # Emit the returned signal with value `result`
        self.signals.returned.emit(success)


class SegmentationDiagnosticsManager(WorkerBase):
    """Runs the SegmentationDiagnostics Tool in a separate Qt thread.

    Extends napari's WorkerBase.
    """

    def __init__(self, segm_diag):
        """Constructor.

        @param segm_diag: obj
            SegmentationDiagnostics tool to be injected.
        """

        # Call base constructor
        super().__init__()

        # Store reference to the Segmentation Diagnostics tool
        self._segm_diag = segm_diag

        # Error message
        self._message = ""

    @pyqtSlot()
    def run(self):
        """Run method to be executed in a separate thread."""

        # Emit started signal
        self.signals.started.emit()

        if self._segm_diag is None:
            # Emit the errored signal with the error message
            self.signals.errored.emit("The Segmentation Diagnostics tool was not properly initialized!")

            plot_path = False

        else:

            # Run the tool
            plot_path = self._segm_diag.evaluate_segmentation()

            if not plot_path:
                # Emit the errored signal with the error message
                self.signals.errored.emit("The Segmentation Diagnostics tool failed!")

        # Emit the finished signal
        self.signals.finished.emit()

        # Emit the returned signal with value `plot_path`
        self.signals.returned.emit(plot_path)
