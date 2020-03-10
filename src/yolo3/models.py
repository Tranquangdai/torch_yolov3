import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from src.yolo3.utils import chain, letterbox_image
from torch import nn


class DarknetConv2D_BN_Leaky(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        if stride == 2:
            padding = 0
        else:
            padding = int(
                ((stride - 1) * in_channels - stride + kernel_size) / 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding, bias=False)
        self.batch_normalization = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, X):
        return chain(X, [self.conv2d, self.batch_normalization, self.leaky_relu])


class ResLayer(nn.Module):

    def __init__(self, num_filters):
        super(ResLayer, self).__init__()
        self.darkconv_1 = DarknetConv2D_BN_Leaky(
            num_filters, num_filters // 2, 1)
        self.darkconv_2 = DarknetConv2D_BN_Leaky(
            num_filters // 2, num_filters, 3)

    def forward(self, X):
        return X + chain(X, nn.ModuleList([self.darkconv_1, self.darkconv_2]))


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResBlock, self).__init__()
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.darkconv = DarknetConv2D_BN_Leaky(in_channels, out_channels, 3, 2)
        self.reslayer = nn.ModuleDict([
            ['reslayer_{}'.format(i), ResLayer(out_channels)]
            for i in range(num_blocks)
        ])

    def forward(self, X):
        X = chain(X, [self.zero_padding, self.darkconv])
        for name in self.reslayer:
            X = self.reslayer[name](X)
        return X


class DarknetBody(nn.Module):

    def __init__(self):
        super(DarknetBody, self).__init__()
        self.preconv = DarknetConv2D_BN_Leaky(3, 32, 3, 1)

        self.resblock_1 = ResBlock(32, 64, 1)
        self.resblock_2 = ResBlock(64, 128, 2)
        self.resblock_3 = ResBlock(128, 256, 8)
        self.resblock_4 = ResBlock(256, 512, 8)
        self.resblock_5 = ResBlock(512, 1024, 4)

    def forward(self, X):
        cache = []
        resblock_3 = chain(X, [
            self.preconv,
            self.resblock_1,
            self.resblock_2,
            self.resblock_3
        ])
        resblock_4 = chain(resblock_3, [self.resblock_4])
        out = chain(resblock_4, [self.resblock_5])
        return resblock_3, resblock_4, out


class LastLayer(nn.Module):

    def __init__(self, in_filters, num_filters, out_filters):
        super(LastLayer, self).__init__()
        self.darkconv_1 = DarknetConv2D_BN_Leaky(in_filters, num_filters, 1)
        self.darkconv_2 = DarknetConv2D_BN_Leaky(
            num_filters, num_filters * 2, 3)
        self.darkconv_3 = DarknetConv2D_BN_Leaky(
            num_filters * 2, num_filters, 1)
        self.darkconv_4 = DarknetConv2D_BN_Leaky(
            num_filters, num_filters * 2, 3)
        self.darkconv_5 = DarknetConv2D_BN_Leaky(
            num_filters * 2, num_filters, 1)

        self.darkconv_6 = DarknetConv2D_BN_Leaky(
            num_filters, num_filters * 2, 3)
        self.conv2d = nn.Conv2d(num_filters * 2, out_filters, 1, 1)

    def forward(self, x):
        x = chain(x, [
            self.darkconv_1,
            self.darkconv_2,
            self.darkconv_3,
            self.darkconv_4,
            self.darkconv_5
        ])
        y = chain(x, [self.darkconv_6, self.conv2d])
        return x, y


class YoloBody(nn.Module):

    def __init__(self, num_anchors, num_classes):
        depth = num_anchors * (num_classes + 5)
        super(YoloBody, self).__init__()
        self.darkbody = DarknetBody()
        self.lastlayer_1 = LastLayer(1024, 512, depth)

        self.darkconv_1 = DarknetConv2D_BN_Leaky(512, 256, 1, 1)
        self.lastlayer_2 = LastLayer(768, 256, depth)

        self.darkconv_2 = DarknetConv2D_BN_Leaky(256, 128, 1, 1)
        self.lastlayer_3 = LastLayer(384, 128, depth)

    def forward(self, x):
        resblock_3, resblock_4, darkout = self.darkbody(x)
        x, y1 = self.lastlayer_1(darkout)

        x = F.interpolate(self.darkconv_1(x), scale_factor=2)
        x = torch.cat((x, resblock_4), dim=1)
        x, y2 = self.lastlayer_2(x)

        x = F.interpolate(self.darkconv_2(x), scale_factor=2)
        x = torch.cat((x, resblock_3), dim=1)
        x, y3 = self.lastlayer_3(x)

        return (y1, y2, y3)
