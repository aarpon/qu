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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
import napari

from qu.ui.qu_main_widget import QuMainWidget


def qu_launcher():

    # Disable HiDPI scaling since it breaks OpenGL-based rendering
    QGuiApplication.setAttribute(Qt.AA_DisableHighDpiScaling)

    with napari.gui_qt():

        # Instantiate napari viewer with custom title
        viewer = napari.Viewer(title="Qu for napari")

        # Instantiate QuMainWidget
        quMainWidget = QuMainWidget(viewer)

        # Add to dock
        viewer.window.add_dock_widget(quMainWidget, name='Qu', area='right')

        # If there is enough space, enlarge the main window to fit all
        # widgets properly
        viewer.window.resize(1600, 1000)
