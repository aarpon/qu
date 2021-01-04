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



