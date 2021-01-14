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
