from PyQt5.QtCore import QObject, pyqtSignal
import sys


class EmittingOutputStream(QObject):
    """
    Implementation of a stream to handle logging messages to a Qt widget.
    """

    """Signal to redirect text."""
    stream_signal = pyqtSignal(str)

    def __init__(self):
        """Constructor."""
        super().__init__()

    def write(self, text: object) -> None:
        """Emits text formatted as string.

        @param text: object
            Any object that can be converted to string via str(text).
        """
        self.stream_signal.emit(str(text))

    def flush(self):
        """Flush the stream."""
        sys.stdout = sys.__stdout__
        sys.stdout.flush()

    def close(self):
        """Close the stream."""
        print("CLOSE WAS CALLED!")
        self.flush()

    def __del__(self):
        """Destructor."""
        sys.stdout = sys.__stdout__


class EmittingErrorStream(QObject):
    """
    Implementation of a stream to handle logging messages to a Qt widget.
    """

    """Signal to redirect text."""
    stream_signal = pyqtSignal(str)

    def __init__(self):
        """Constructor."""
        super().__init__()

    def write(self, text: object) -> None:
        """Emits text formatted as string.

        @param text: object
            Any object that can be converted to string via str(text).
        """
        self.stream_signal.emit(str(text))

    def flush(self):
        """Flush the stream."""
        sys.stderr = sys.__stderr__
        sys.stderr.flush()

    def close(self):
        """Close the stream."""
        print("CLOSE WAS CALLED!")
        self.flush()

    def __del__(self):
        """Destructor."""
        sys.stderr = sys.__stderr__
