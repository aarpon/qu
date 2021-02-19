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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicUNet2D(nn.Module):
    """U-Net 2D implementation in PyTorch.

    Heavily inspired from https://github.com/jvanvugt/pytorch-unet.

    Using the default parameters will implement the version of U-Net
    from the original publication:

    @misc{ronneberger2015unet,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

    @see https://arxiv.org/abs/1505.04597
    @see https://github.com/jvanvugt/pytorch-unet
    @see https://www.manning.com/books/deep-learning-with-pytorch
    """

    def __init__(
            self,
            in_channels: int = 1,
            n_classes: int = 2,
            depth: int = 5,
            wf: int = 6,
            padding: bool = False,
            batch_norm: bool = False,
            up_mode: str = 'upconv'
    ):
        """Constructor.
        :param in_channels: int
            Number of input channels.

        :param n_classes: int
            Number of output classes.

        :param depth: int
            Depth of the network

        :param wf: int
            Defines the number of filters in the first layer as 2**wf.

        :param padding: bool
            Toggle padding of the input image for convolution. Set to
            True to make sure the output has the same size as the input.

        :param batch_norm: bool
            Uses batch normalizations after layers with an activation function.

        :param up_mode: str
            Up-sampling mode. One of 'upconv' or 'upsample'.
            'upconv': use transposed convolutions for learned upsampling.
            'upsample': use bilinear upsampling.
        """

        # Call base constructor
        super().__init__()

        # Check the up-sample mode
        if up_mode not in ('upconv', 'upsample'):
            raise ValueError("Bad value for up_mode.")

        # Store the arguments
        self.padding = padding
        self.depth = depth

        # Since the kernel size is fixed at three, we also
        # hard-code the padding size
        self.int_padding = 1 if padding else 0

        # Keep track of the last number of input channels
        prev_channels = in_channels

        # Build the down path
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                _UNet2DConvBlock(
                    in_size=prev_channels,
                    out_size=2 ** (wf + i),
                    padding=self.int_padding,
                    batch_norm=batch_norm
                )
            )
            prev_channels = 2 ** (wf + i)

        # Build the up path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                _UNet2DUpBlock(
                    in_size=prev_channels,
                    out_size=2 ** (wf + i),
                    up_mode=up_mode,
                    padding=self.int_padding,
                    batch_norm=batch_norm
                )
            )
            prev_channels = 2 ** (wf + i)

        # Last layer
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """Implement the forward path."""

        blocks = []

        # Go down the down_path
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        # Go up the up_path
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        # Apply the last 2D convolution and return
        return self.last(x)


class _UNet2DConvBlock(nn.Module):
    """U-Net convolution block."""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 padding: int,
                 batch_norm: bool
                 ):
        """Constructor.

        :param in_size: int
            Number of input channels.

        :param out_size: int
            Number of output channels.

        :param padding: int
            Border padding for convolution.

        :param batch_norm: bool
            Uses batch normalizations after layers with an activation function.
        """

        # Call base constructor
        super().__init__()

        # Prepare the list of blocks
        block = []

        # Add a Conv2d module
        block.append(
            nn.Conv2d(
                in_size,
                out_size,
                kernel_size=3,
                padding=padding
            )
        )

        # Add a ReLU activation
        block.append(nn.ReLU())

        # If requested, append batch normalization
        if batch_norm:
            block.append(
                nn.BatchNorm2d(out_size)
            )

        # Add another Conv2d module
        block.append(
            nn.Conv2d(
                out_size,
                out_size,
                kernel_size=3,
                padding=padding
            )
        )

        # Add a ReLU activation
        block.append(nn.ReLU())

        # If requested, append batch normalization
        if batch_norm:
            block.append(
                nn.BatchNorm2d(out_size)
            )

        # Turn the list into an nn.Sequential()
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Implement forward pass.

        @param x: tensor
            Input.
        """

        out = self.block(x)
        return out


class _UNet2DUpBlock(nn.Module):
    """U-Net up block."""

    def __init__(
            self,
            in_size: int,
            out_size: int,
            up_mode: str,
            padding: int,
            batch_norm: bool
    ):
        """Constructor.

        :param in_size: int
            Number of input channels.

        :param out_size: int
            Number of output channels.

        :param up_mode: str
            Up-sampling mode. One of 'upconv' or 'upsample'.
            'upconv': use transposed convolutions for learned upsampling.
            'upsample': use bilinear upsampling.

        :param padding: int
            Border padding for convolution.

        :param batch_norm: bool
            Uses batch normalizations after layers with an activation function.

        """

        # Call to base constructor
        super().__init__()

        # Check the up-sample mode
        if up_mode not in ('upconv', 'upsample'):
            raise ValueError("Bad value for up_mode.")

        # Prepare the list of blocks
        block = []

        # Add upsampling
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(
                in_size,
                out_size,
                kernel_size=2,
                stride=2
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(
                    mode="bilinear",
                    scale_factor=2,
                    align_corners=False
                ),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=1
                )
            )

        self.conv_block = _UNet2DConvBlock(
            in_size=in_size,
            out_size=out_size,
            padding=padding,
            batch_norm=batch_norm
        )

    def center_crop(self, layer, target_size):
        """Return the center area or the layer.

        @param: layer: tensor
            Layer to be cropped.

        @param target_size: tuple(samples, channels, height, width)
            Size of the cropped layer.
        """

        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :,
               diff_y:(diff_y + target_size[0]),
               diff_x:(diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        """Forward pass.

        @param x: tensor
            Input.

        @param bridge: tensor
             Tensor for skip connection.
        """

        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out
