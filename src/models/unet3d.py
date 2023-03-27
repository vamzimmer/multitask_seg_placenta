import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
    Unet model for segmentation

    Unet3D
    ResUnet3D

"""


class ConvSingle(nn.Module):
    def __init__(self, inChannel, outChannel, groupChannel=32, kernel_size=3, padding=1):
        super(ConvSingle, self).__init__()
        groups = min(outChannel, groupChannel)
        self.Conv = nn.Sequential(
            nn.Conv3d(inChannel, outChannel, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.GroupNorm(groups, outChannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Conv(x)
        return x


class ConvDouble(nn.Module):
    def __init__(self, inChannel, outChannel, hasDropout=False, dropout_rate=0.2, groupChannel=32):
        super(ConvDouble, self).__init__()
        groups = min(outChannel, groupChannel)
        self.hasDropout = hasDropout
        self.inplace = not self.hasDropout
        self.Conv1 = nn.Sequential(
            nn.Conv3d(inChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(groups, outChannel),
            nn.ReLU(inplace=self.inplace)
        )
        # self.Dropout = nn.Dropout2d(0.2, True)
        self.Dropout = nn.Dropout3d(p=dropout_rate, inplace=True)
        self.Conv2 = nn.Sequential(
            nn.Conv3d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(groups, outChannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Conv1(x)
        if self.hasDropout:
            # print(self.hasDropout)
            x = self.Dropout(x)
        x = self.Conv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel, groupChannel=32, dropout=False, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.Conv_1x1 = nn.Conv3d(inChannel, outChannel, kernel_size=1, stride=1, padding=0)
        self.Conv = ConvDouble(outChannel, outChannel, groupChannel=groupChannel, hasDropout=dropout,
                               dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.Conv(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(LinearBlock, self).__init__()
        self.Linear = nn.Sequential(
            nn.Linear(in_features=inFeatures, out_features=outFeatures),
            nn.LayerNorm([outFeatures]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Linear(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inChannel, outChannel, groupChannel=32, dropout=False, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        self.Conv_1x1 = nn.Conv3d(inChannel, outChannel, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Sequential(
            ConvDouble(outChannel, outChannel, groupChannel=groupChannel, hasDropout=dropout, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        # print(x.size())
        x = self.Conv_1x1(x)
        # print(x.size())
        x1 = self.Conv(x)
        # print(x1.size())
        return x + x1


class UNet3D(nn.Module):
    def __init__(self, inchannels, outchannels=2, first_channels=32, image_size=(256, 256),
                 levels=5, dropout=False, dropout_rate=0.2, concatenation=False):
        super(UNet3D, self).__init__()
        channels = first_channels
        self.levels = levels
        self.dropout = dropout
        self.kernel_size = 3
        self.stride = 2
        self.image_size = image_size
        self.concatenation = concatenation
        concat_factor = 1 if not self.concatenation else 2

        self.n_channels = (inchannels,)
        for i in range(0, self.levels):
            self.n_channels = self.n_channels + (channels * pow(2, i),)
        print(self.n_channels)

        # Define the convolutional layers
        self.layer_sizes = (list(self.image_size),)
        for idx in range(self.levels - 1):
            newsize = [np.floor(
                (s + 2 * np.floor((self.kernel_size - 1) / 2) - self.kernel_size) / self.stride + 1).astype(np.int) for
                       s in self.layer_sizes[idx]]
            self.layer_sizes = self.layer_sizes + (newsize,)
        print(self.layer_sizes)

        # encoder
        encoders = [ConvBlock(self.n_channels[0], self.n_channels[1], groupChannel=first_channels, dropout=self.dropout,
                              dropout_rate=dropout_rate)]
        for i in range(1, self.levels):
            block = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ConvBlock(self.n_channels[i], self.n_channels[i + 1], groupChannel=first_channels, dropout=self.dropout,
                          dropout_rate=dropout_rate)
            )
            encoders.append(block)
        self.encoders = nn.ModuleList(encoders)

        # create decoder levels
        decoders = [nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvSingle(self.n_channels[self.levels], self.n_channels[self.levels - 1], groupChannel=first_channels)
            )]
        for i in range(self.levels - 1, 1, -1):
            block = nn.Sequential(
                ConvBlock(concat_factor*self.n_channels[i], self.n_channels[i], groupChannel=first_channels),
                nn.Upsample(scale_factor=2),
                ConvSingle(self.n_channels[i], self.n_channels[i-1], groupChannel=first_channels)
            )
            decoders.append(block)
        decoders.append(ConvBlock(concat_factor*self.n_channels[1], self.n_channels[1], groupChannel=first_channels))
        self.decoders = nn.ModuleList(decoders)

        # self.lastConv = ConvSingle(self.n_channels[1], outchannels)
        self.lastConv = nn.Conv3d(self.n_channels[1], outchannels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):


        inputStack = []
        l_i = x
        for i in range(self.levels):
            l_i = self.encoders[i](l_i)
            if i < self.levels - 1:
                inputStack.append(l_i)

        x = l_i
        for i in range(self.levels):

            x = self.decoders[i](x)
            if i < self.levels - 1:
                if not self.concatenation:
                    x = x + inputStack.pop()
                else:
                    x = torch.cat([inputStack.pop(), x], 1)

        x = self.lastConv(x)
        x = torch.sigmoid(x)

        return x


class ResUNet3D(nn.Module):
    def __init__(self, inchannels, outchannels=2, first_channels=32, image_size=(256, 256),
                 levels=5, dropout=False, dropout_dec=False, dropout_rate=0.2, concatenation=False):
        super(ResUNet3D, self).__init__()
        channels = first_channels
        self.levels = levels
        self.dropout = dropout
        self.dropout_dec = dropout_dec
        self.kernel_size = 3
        self.stride = 2
        self.image_size = image_size
        self.concatenation = concatenation
        concat_factor = 1 if not self.concatenation else 2

        self.n_channels = (inchannels,)
        for i in range(0, self.levels):
            self.n_channels = self.n_channels + (channels * pow(2, i),)
        print(self.n_channels)

        # Define the convolutional layers
        self.layer_sizes = (list(self.image_size),)
        for idx in range(self.levels - 1):
            newsize = [np.floor(
                (s + 2 * np.floor((self.kernel_size - 1) / 2) - self.kernel_size) / self.stride + 1).astype(np.int) for
                       s in self.layer_sizes[idx]]
            self.layer_sizes = self.layer_sizes + (newsize,)
        print(self.layer_sizes)

        # encoder
        encoders = [ResidualBlock(self.n_channels[0], self.n_channels[1], groupChannel=first_channels,
                                  dropout=self.dropout, dropout_rate=dropout_rate)]
        for i in range(1, self.levels):
            block = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ResidualBlock(self.n_channels[i], self.n_channels[i + 1], groupChannel=first_channels,
                              dropout=self.dropout, dropout_rate=dropout_rate)
            )
            encoders.append(block)
        self.encoders = nn.ModuleList(encoders)

        # create decoder levels
        decoders = [nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvSingle(self.n_channels[self.levels], self.n_channels[self.levels - 1], groupChannel=first_channels)
            )]
        for i in range(self.levels - 1, 1, -1):
            block = nn.Sequential(
                ResidualBlock(concat_factor*self.n_channels[i], self.n_channels[i], groupChannel=first_channels,
                              dropout=self.dropout_dec, dropout_rate=dropout_rate),
                nn.Upsample(scale_factor=2),
                # add a convolution layer here to reduce the channel number.
                ConvSingle(self.n_channels[i], self.n_channels[i-1], groupChannel=first_channels)
            )
            decoders.append(block)
        decoders.append(ResidualBlock(concat_factor*self.n_channels[1], self.n_channels[1], groupChannel=first_channels,
                                      dropout=self.dropout_dec, dropout_rate=dropout_rate))
        self.decoders = nn.ModuleList(decoders)
        # print("Decoder length {}".format(len(decoders)))

        # self.lastConv = ConvSingle(self.n_channels[1], outchannels)
        self.lastConv = nn.Conv3d(self.n_channels[1], outchannels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        inputStack = []
        l_i = x
        for i in range(self.levels):
            l_i = self.encoders[i](l_i)
            if i < self.levels - 1:
                inputStack.append(l_i)

        x = l_i
        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                if not self.concatenation:
                    x = x + inputStack.pop()
                else:
                    x = torch.cat([inputStack.pop(), x], 1)

        x = self.lastConv(x)
        x = torch.sigmoid(x)


        return x
