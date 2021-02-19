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
#   Taken (with slight modifications) from:
#   https://towardsdatascience.com/biomedical-image-segmentation-attention-u-net-29b6f0827405

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class _UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class _AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(_AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttentionUNet2D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, n1=64):
        super(AttentionUNet2D, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = _ConvBlock(img_ch, filters[0])
        self.Conv2 = _ConvBlock(filters[0], filters[1])
        self.Conv3 = _ConvBlock(filters[1], filters[2])
        self.Conv4 = _ConvBlock(filters[2], filters[3])
        self.Conv5 = _ConvBlock(filters[3], filters[4])

        self.Up5 = _UpConv(filters[4], filters[3])
        self.Att5 = _AttentionBlock(f_g=filters[3], f_l=filters[3], f_int=filters[2])
        self.Up_conv5 = _ConvBlock(filters[4], filters[3])

        self.Up4 = _UpConv(filters[3], filters[2])
        self.Att4 = _AttentionBlock(f_g=filters[2], f_l=filters[2], f_int=filters[1])
        self.Up_conv4 = _ConvBlock(filters[3], filters[2])

        self.Up3 = _UpConv(filters[2], filters[1])
        self.Att3 = _AttentionBlock(f_g=filters[1], f_l=filters[1], f_int=filters[0])
        self.Up_conv3 = _ConvBlock(filters[2], filters[1])

        self.Up2 = _UpConv(filters[1], filters[0])
        self.Att2 = _AttentionBlock(f_g=filters[0], f_l=filters[0], f_int=32)
        self.Up_conv2 = _ConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.MaxPool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out
