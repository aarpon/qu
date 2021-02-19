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

import os
from dataclasses import dataclass
from typing import Tuple

from qu.models.core import Architecture


@dataclass
class UNet2DSegmenterSettings:
    """Settings for the UNet segmenter."""

    # Architecture
    architecture: Architecture = Architecture.ResidualUNet2D

    # Number of epochs
    num_epochs: int = 10

    # Validation step
    validation_step: int = 2

    # Batch sizes
    batch_sizes: Tuple[int, int, int, int] = (8, 1, 1, 1)

    # ROI size (height, width)
    roi_size: Tuple[int, int] = (384, 384)

    # Number of filters in the first layers
    num_filters_in_first_layer: int = 16

    # Number of workers
    num_workers: Tuple[int, int, int, int] = (
        os.cpu_count(),
        os.cpu_count(),
        os.cpu_count(),
        os.cpu_count()
    )

    # Sliding window batch size
    sliding_window_batch_size: int = 4

    def to_dict(self):
        """Return settings as a dictionary."""
        return {
            "architecture": self.architecture,
            "num_epochs": self.num_epochs,
            "validation_step": self.validation_step,
            "batch_sizes": self.batch_sizes,
            "roi_size": self.roi_size,
            "num_filters_in_first_layer": self.num_filters_in_first_layer,
            "num_workers": self.num_workers,
            "sliding_window_batch_size": self.sliding_window_batch_size
        }
