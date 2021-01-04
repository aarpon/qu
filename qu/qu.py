import napari

from qu.ui.qu_main_widget import QuWidget


def qu_launcher():

    with napari.gui_qt():

        # Instantiate napari viewer
        viewer = napari.Viewer()

        # Instantiate QuWidget
        quWidget = QuWidget(viewer)

        # Add to dock
        viewer.window.add_dock_widget(quWidget, name='Qu', area='right')

        # If there is enough space, enlarge the main window to fit all
        # widgets properly
        viewer.window.resize(1600, 800)
