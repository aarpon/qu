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


@dataclass
class UNet2DRestorerSettings:
    """Settings for the UNet Restorer."""

    # Global intensity minimum for normalization
    norm_min: int = 0

    # Global intensity maximum for normalization
    norm_max: int = 65535

    # Number of samples per images
    num_samples: int = 1

    # Number of epochs
    num_epochs: int = 10

    # Validation step
    validation_step: int = 2

    # Batch sizes
    batch_sizes: Tuple[int, int, int, int] = (8, 1, 1, 1)

    # ROI size (height, width)
    roi_size: Tuple[int, int] = (128, 128)

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
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
            "num_samples": self.num_samples,
            "num_epochs": self.num_epochs,
            "validation_step": self.validation_step,
            "batch_sizes": self.batch_sizes,
            "roi_size": self.roi_size,
            "num_workers": self.num_workers,
            "sliding_window_batch_size": self.sliding_window_batch_size
        }
