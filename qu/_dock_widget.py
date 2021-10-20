"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation

from qu.ui.qu_main_widget import QuMainWidget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():

    # Return the QuMainWidget class
    return QuMainWidget
