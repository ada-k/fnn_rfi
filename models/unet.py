"""Small UNet implementation for 2D spectrogram segmentation.

Designed as a lightweight baseline for RFI segmentation tasks. Includes
commented, configurable blocks and a final 1-channel mask output.
"""
from typing import List

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        return self.pool(y), y


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # concat along channel dim
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, depth=4):
        super().__init__()
        # encoder
        self.downs = nn.ModuleList()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        prev = in_ch
        for c in chs:
            self.downs.append(Down(prev, c))
            prev = c

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNAct(prev, prev * 2),
            ConvBNAct(prev * 2, prev * 2),
        )

        # decoder
        self.ups = nn.ModuleList()
        rev = list(reversed(chs))
        prev = prev * 2
        for c in rev:
            self.ups.append(Up(prev, c))
            prev = c

        # final conv to single-channel mask
        self.final = nn.Conv2d(prev, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x, y = down(x)
            skips.append(y)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        x = self.final(x)
        return x


if __name__ == "__main__":
    m = UNet(in_ch=1, base_ch=8, depth=3)
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    print(y.shape)
