import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UNet2DSegmenterSettings:
    """Settings for the UNet segmenter."""

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

    # Sliding window batch size
    sliding_window_batch_size: int = 4

    def to_dict(self):
        """Return settings as a dictionary."""
        return {
            "num_epochs": self.num_epochs,
            "validation_step": self.validation_step,
            "batch_sizes": self.batch_sizes,
            "roi_size": self.roi_size,
            "num_workers": self.num_workers,
            "sliding_window_batch_size": self.sliding_window_batch_size
        }
