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

import unittest
from pathlib import Path

import numpy as np

from qu.data import DataManager, ExperimentType, MaskType


class DataManagerTestCase(unittest.TestCase):

    def test_inferring_classification_experiment(self):

        # Data folder
        data_folder = Path(__file__).parent / "data" / "demo_classification"

        # Instantiate DataManager
        dataManager = DataManager()
        dataManager.root_data_path = data_folder

        # Make sure the experiment type was inferred correctly
        self.assertEqual(ExperimentType.CLASSIFICATION, dataManager.experiment_type)

        # Scan the data
        result = dataManager.scan()
        self.assertEqual(True, result)

        # Get info about the data
        self.assertEqual(1, dataManager.num_input_channels)
        self.assertEqual(3, dataManager.num_classes)
        self.assertEqual(3, dataManager.num_output_channels)
        self.assertEqual(3, dataManager.num_images)
        self.assertEqual(3, dataManager.num_masks)
        self.assertEqual(0, dataManager.num_targets)
        self.assertEqual(MaskType.TIFF_LABELS, dataManager.mask_type)

        # Check the returned data
        image, mask = dataManager.get_image_data_at_current_index()

        # Check image
        self.assertEqual(np.uint16, image.dtype)
        self.assertEqual(2, image.ndim)

        # Check mask
        self.assertEqual(np.int32, mask.dtype)
        self.assertEqual(2, image.ndim)
        self.assertEqual(3, len(np.unique(mask)))

    def test_inferring_regression_experiment(self):

        # Data folder
        data_folder = Path(__file__).parent / "data" / "demo_regression"

        # Instantiate DataManager
        dataManager = DataManager()
        dataManager.root_data_path = data_folder

        # Make sure the experiment type was inferred correctly
        self.assertEqual(ExperimentType.REGRESSION, dataManager.experiment_type)

        # Scan the data
        result = dataManager.scan()
        self.assertEqual(True, result)

        # Get info about the data
        self.assertEqual(1, dataManager.num_input_channels)
        self.assertEqual(0, dataManager.num_classes)
        self.assertEqual(1, dataManager.num_output_channels)
        self.assertEqual(3, dataManager.num_images)
        self.assertEqual(0, dataManager.num_masks)
        self.assertEqual(3, dataManager.num_targets)

        # Check the returned data
        image, target = dataManager.get_image_data_at_current_index()

        # Check image
        self.assertEqual(np.uint16, image.dtype)
        self.assertEqual(2, image.ndim)

        # Check mask
        self.assertEqual(np.uint16, target.dtype)
        self.assertEqual(2, image.ndim)


if __name__ == '__main__':
    unittest.main()
