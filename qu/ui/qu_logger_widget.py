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

from PyQt5.QtWidgets import QTextEdit


class QuLoggerWidget(QTextEdit):

    def __init__(self, viewer, *args, **kwargs):
        """Constructor."""

        # Call base constructor
        super().__init__(*args, **kwargs)

        # Configure the widget
        self.setReadOnly(True)
        self.setAcceptDrops(False)
