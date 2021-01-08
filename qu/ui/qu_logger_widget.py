from PyQt5.QtWidgets import QTextEdit


class QuLoggerWidget(QTextEdit):

    def __init__(self, viewer, *args, **kwargs):
        """Constructor."""

        # Call base constructor
        super().__init__(*args, **kwargs)

        # Configure the widget
        self.setReadOnly(True)
        self.setAcceptDrops(False)
