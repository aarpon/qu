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
#

import dataclasses
import math

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QDialog

from qu.models.core import Architectures, Losses, Optimizers
from qu.ui import _ui_folder_path


class QuUNetSegmenterSettingsDialog(QDialog):
    """Dialog to edit the settings of the UNetLearner.

    A copy of the passed settings is edited and returned if
    the Accept button is pressed; otherwise, None. The original
    settings are unaffected.
    """

    def __init__(self, settings, *args, **kwargs):
        """Constructor."""

        # Call base constructor
        super().__init__(*args, **kwargs)

        # Store a reference to __a copy__ of the settings
        self._settings = dataclasses.replace(settings)

        # Set up UI
        uic.loadUi(_ui_folder_path / "dialogs" / "qu_unet_segmenter_settings_dialog.ui", self)

        # Make sure to set validators where necessary
        self._set_validators()

        # Fill the fields
        self._fill_ui_fields()

        # Set the connections
        self._set_connections()

    def _set_validators(self):
        """Set validators on edit fields."""
        self.leNumEpochs.setValidator(QIntValidator(1, 1000000000, self))
        self.leLearningRate.setValidator(QDoubleValidator(0.0, 1.0, 8, self))
        self.leWeightDecay.setValidator(QDoubleValidator(0.0, 1.0, 8, self))
        self.leMomentum.setValidator(QDoubleValidator(0.0, 1.0, 8, self))
        self.leValidationStep.setValidator(QIntValidator(1, 1000000000, self))
        self.leTrainingBatchSize.setValidator(QIntValidator(1, 1000000000, self))
        self.lineEditROIHeight.setValidator(QIntValidator(1, 1000000000, self))
        self.lineEditROIWidth.setValidator(QIntValidator(1, 1000000000, self))
        self.leNumWorkers.setValidator(QIntValidator(0, 1000000000, self))
        self.leSlidingWindowBatchSize.setValidator(QIntValidator(1, 1000000000, self))

    def _fill_ui_fields(self):
        """Fill the UI elements with the values in the settings."""
        self.cbArchitecture.setCurrentIndex(self._settings.architecture.value)
        self.cbLoss.setCurrentIndex(self._settings.loss.value)
        self.cbOptimizer.setCurrentIndex(self._settings.optimizer.value)
        self.leLearningRate.setText(str(self._settings.learning_rate))
        self.leWeightDecay.setText(str(self._settings.weight_decay))
        self.leMomentum.setText(str(self._settings.momentum))
        self.leNumEpochs.setText(str(self._settings.num_epochs))
        self.cbNumFiltersInFirstLayer.setCurrentIndex(int(math.log2(self._settings.num_filters_in_first_layer) - 1))
        self.leValidationStep.setText(str(self._settings.validation_step))
        self.leTrainingBatchSize.setText(str(self._settings.batch_sizes[0]))
        self.lineEditROIHeight.setText(str(self._settings.roi_size[0]))
        self.lineEditROIWidth.setText(str(self._settings.roi_size[1]))
        self.leNumWorkers.setText(str(self._settings.num_workers[0]))
        self.leSlidingWindowBatchSize.setText(str(self._settings.sliding_window_batch_size))

    def _set_connections(self):
        """Plug signals and slots"""
        self.cbArchitecture.currentIndexChanged.connect(self._on_architecture_value_changed)
        self.cbLoss.currentIndexChanged.connect(self._on_loss_value_changed)
        self.cbOptimizer.currentIndexChanged.connect(self._on_optimizer_value_changed)
        self.leLearningRate.textChanged.connect(self._on_learning_rate_text_changed)
        self.leWeightDecay.textChanged.connect(self._on_weight_decay_text_changed)
        self.leMomentum.textChanged.connect(self._on_momentum_text_changed)
        self.leNumEpochs.textChanged.connect(self._on_num_epochs_text_changed)
        self.cbNumFiltersInFirstLayer.currentIndexChanged.connect(self._on_num_filters_in_first_layer_changed)
        self.leValidationStep.textChanged.connect(self._on_validation_step_text_changed)
        self.leTrainingBatchSize.textChanged.connect(self._on_training_batch_size_text_changed)
        self.lineEditROIHeight.textChanged.connect(self._on_roi_height_text_changed)
        self.lineEditROIWidth.textChanged.connect(self._on_roi_width_text_changed)
        self.leNumWorkers.textChanged.connect(self._on_num_workers_text_changed)
        self.leSlidingWindowBatchSize.textChanged.connect(self._on_sliding_window_batch_size_text_changed)

    @staticmethod
    def get_settings(settings, parent=None):
        """Static method to create the dialog and return the updated settings or None if cancelled."""
        dialog = QuUNetSegmenterSettingsDialog(settings, parent)
        if QDialog.Accepted == dialog.exec_():
            return dialog._settings
        return None

    @pyqtSlot(int, name="_on_architecture_value_changed")
    def _on_architecture_value_changed(self, value):
        """Architecure."""
        if value == 0:
            self._settings.architecture = Architectures.ResidualUNet2D
        elif value == 1:
            self._settings.architecture = Architectures.AttentionUNet2D
        else:
            raise ValueError(f"Unexpected option {value} for architecture.")

    @pyqtSlot(int, name="_on_loss_value_changed")
    def _on_loss_value_changed(self, value):
        """Loss function."""
        if value == 0:
            self._settings.loss = Losses.GeneralizedDiceLoss
        else:
            raise ValueError(f"Unexpected option {value} for loss.")

    @pyqtSlot(int, name="_on_optimizer_value_changed")
    def _on_optimizer_value_changed(self, value):
        """Optimizer."""
        if value == 0:
            self._settings.optimizer = Optimizers.Adam
        elif value == 1:
            self._settings.optimizer = Optimizers.SGD
        else:
            raise ValueError(f"Unexpected option {value} for optimizer.")

    @pyqtSlot('QString', name="_on_learning_rate_text_changed")
    def _on_learning_rate_text_changed(self, str_value):
        """Learning rate."""
        # The DoubleValidator allows empty strings.
        if str_value == '':
            return
        value = float(str_value)
        if value < 0.0:
            value = 0.0
            self.leLearningRate.setText(str(value))
        if value > 1.0:
            value = 1.0
            self.leLearningRate.setText(str(value))
        self._settings.learning_rate = value

    @pyqtSlot('QString', name="_on_weight_decay_text_changed")
    def _on_weight_decay_text_changed(self, str_value):
        """Learning rate."""
        # The DoubleValidator allows empty strings.
        if str_value == '':
            return
        value = float(str_value)
        if value < 0.0:
            value = 0.0
            self.leWeightDecay.setText(str(value))
        if value > 1.0:
            value = 1.0
            self.leWeightDecay.setText(str(value))
        self._settings.weight_decay = value

    @pyqtSlot('QString', name="_on_momentum_text_changed")
    def _on_momentum_text_changed(self, str_value):
        """Learning rate."""
        # The DoubleValidator allows empty strings.
        if str_value == '':
            return
        value = float(str_value)
        if value < 0.0:
            value = 0.0
            self.leMomentum.setText(str(value))
        if value > 1.0:
            value = 1.0
            self.leMomentum.setText(str(value))
        self._settings.momentum = value

    @pyqtSlot('QString', name="_on_num_epochs_text_changed")
    def _on_num_epochs_text_changed(self, str_value):
        """Number of epochs."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.leNumEpochs.setText(str(value))
        self._settings.num_epochs = value

    @pyqtSlot(int, name="_on_num_filters_in_first_layer_changed")
    def _on_num_filters_in_first_layer_changed(self, value):
        """Num filters in first layer."""
        # Change he value to the corresponding power of 2
        self._settings.num_filters_in_first_layer = 2**(value + 1)

    @pyqtSlot('QString', name="_on_validation_step_text_changed")
    def _on_validation_step_text_changed(self, str_value):
        """Validation step."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.leValidationStep.setText(str(value))
        self._settings.validation_step = value

    @pyqtSlot('QString', name="_on_training_batch_size_text_changed")
    def _on_training_batch_size_text_changed(self, str_value):
        """Training batch size step."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.leTrainingBatchSize.setText(str(value))
        new_batch_sizes = (
            value,
            self._settings.batch_sizes[1],
            self._settings.batch_sizes[2],
            self._settings.batch_sizes[3]
        )
        self._settings.batch_sizes = new_batch_sizes

    @pyqtSlot('QString', name="_on_roi_height_text_changed")
    def _on_roi_height_text_changed(self, str_value):
        """ROI height."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.lineEditROIHeight.setText(str(value))
        new_roi_size = (
            value,
            self._settings.roi_size[1]
        )
        self._settings.roi_size = new_roi_size

    @pyqtSlot('QString', name="_on_roi_width_text_changed")
    def _on_roi_width_text_changed(self, str_value):
        """ROI width."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.lineEditROIWidth.setText(str(value))
        new_roi_size = (
            self._settings.roi_size[0],
            value
        )
        self._settings.roi_size = new_roi_size

    @pyqtSlot('QString', name="_on_num_workers_text_changed")
    def _on_num_workers_text_changed(self, str_value):
        """Number of workers.

        For now, we set the same number of workers for
        training, validation, testing, and prediction.
        """
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 0:
            value = 0
            self.leNumWorkers.setText(str(value))
        new_num_workers = (
            value,
            value,
            value,
            value
        )
        self._settings.num_workers = new_num_workers

    @pyqtSlot('QString', name="_on_sliding_window_batch_size_text_changed")
    def _on_sliding_window_batch_size_text_changed(self, str_value):
        """Sliding window batch size (for prediction)."""
        # The IntValidator allows empty strings.
        if str_value == '':
            return
        value = int(str_value)
        if value < 1:
            value = 1
            self.sliding_window_batch_size.setText(str(value))
        self._settings.sliding_window_batch_size = value
