import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UNet2DSettings:
    """Settings for the UNet learner."""

    # Number of epochs
    num_epochs: int = 10

    # Validation step
    validation_step: int = 2

    # Batch sizes
    batch_sizes: Tuple[int, int, int, int] = (8, 1, 1, 1)

    # ROI size (height, width)
    roi_size: Tuple[int, int] = (384, 384)

    # Number of workers
    num_workers: Tuple[int, int, int, int] = (
        os.cpu_count(),
        os.cpu_count(),
        os.cpu_count(),
        os.cpu_count()
    )
