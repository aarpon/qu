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
        self.flush()

    def __del__(self):
        """Destructor."""
        sys.stdout = sys.__stdout__

    def isatty(self):
        """Override isatty() method."""
        return True


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
        self.flush()

    def __del__(self):
        """Destructor."""
        sys.stderr = sys.__stderr__

    def isatty(self):
        """Override isatty() method."""
        return True
