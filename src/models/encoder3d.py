import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
    Encoder model for classification with attention layers [1]

    EncoderAtt
    ResEncoderAtt

    [1] Jetley, Saumya, et al. "Learn to pay attention." arXiv preprint arXiv:1804.02391 (2018).
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

class ConvDouble(nn.Module):
    def __init__(self, inChannel, outChannel, hasDropout=False, groupChannel=32, dropout_rate=0.2):
        super(ConvDouble, self).__init__()
        groups = min(outChannel, groupChannel)
        self.hasDropout = hasDropout
        self.inplace = not self.hasDropout
        self.Conv1 = nn.Sequential(
            nn.Conv3d(inChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(groups, outChannel),
            nn.ReLU(inplace=self.inplace)
        )
        self.Dropout = nn.Dropout3d(p=dropout_rate, inplace=True)
        self.Conv2 = nn.Sequential(
            nn.Conv3d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(groups, outChannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Conv1(x)
        if self.hasDropout:
            x = self.Dropout(x)
        x = self.Conv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel, groupChannel=32, dropout=False, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.Conv_1x1 = nn.Conv3d(inChannel, outChannel, kernel_size=1, stride=1, padding=0)
        self.Conv = ConvDouble(outChannel, outChannel, groupChannel=groupChannel, hasDropout=dropout, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.Conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inChannel, outChannel, groupChannel=32, dropout=False, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        self.Conv_1x1 = nn.Conv3d(inChannel, outChannel, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Sequential(
            ConvDouble(outChannel, outChannel, groupChannel=groupChannel, hasDropout=dropout, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.Conv(x)
        return x + x1


class MyLinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(MyLinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.compatibility_score = nn.Conv3d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0,
                                             bias=False)

    def forward(self, l, g):
        N, C, W, H, D = l.size()
        c = self.compatibility_score(l + g) 

        c_min_per_batch = \
            torch.min(torch.min(torch.min(torch.min(c, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0],
                                dim=3, keepdim=True)[0], dim=4, keepdim=True)[0]
        c_m_min = c - c_min_per_batch
        c_sum_per_batch = torch.sum(torch.sum(torch.sum(torch.sum(c_m_min, dim=1, keepdim=True), dim=2, keepdim=True),
                                              dim=3, keepdim=True), dim=4, keepdim=True)
        attention_coef = c_m_min / c_sum_per_batch

        g = torch.mul(attention_coef.expand_as(l), l)
        g = g.view(N, C, -1).sum(dim=2)  

        return attention_coef.view(N, 1, W, H, D), g


class EncoderAtt(nn.Module):
    def __init__(self, inchannels, n_classes, outchannels=2, first_channels=32, image_size=(256, 256, 256),
                 levels=5, dropout=False, dropout_rate=0.2, concatenation=False,
                 attention_layers_pos=(1, 2)):
        super(EncoderAtt, self).__init__()
        channels = first_channels
        self.N_CLASSES = n_classes
        self.levels = levels
        self.dropout = dropout
        self.kernel_size = 3
        self.stride = 2
        self.image_size = image_size
        self.concatenation = concatenation
        self.attention_layers_pos = attention_layers_pos
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

        #
        #   Attention layers
        #
        self.attention_layers = nn.ModuleList(
            [MyLinearAttentionBlock(in_features=self.n_channels[-1], normalize_attn=True) for i in
             self.attention_layers_pos])

        projection = [
            nn.Conv3d(in_channels=self.n_channels[i + 1], out_channels=self.n_channels[-1], kernel_size=1, padding=0,
                      bias=False) for i in self.attention_layers_pos]
        self.projectors_layers = nn.ModuleList(
            projection
        )

        self.upsampler = nn.Upsample(size=tuple(self.image_size), mode='trilinear', align_corners=True)

        #
        #   Classification
        #
        self.dense = ConvSingle(self.n_channels[-1], self.n_channels[-1], kernel_size=self.layer_sizes[-1], padding=0)
        self.classify = LinearBlock(self.n_channels[-1] * len(self.attention_layers_pos), self.N_CLASSES)

    def forward(self, x):

        l_all = []
        l_i = x
        for i in range(self.levels):
            l_i = self.encoders[i](l_i)
            l_all.append(l_i)

        g = self.dense(l_i)

        attentions = []
        gs = []
        for i, layer_pos in enumerate(self.attention_layers_pos):
            p = self.projectors_layers[i](l_all[layer_pos])
            a, g_i = self.attention_layers[i](p, g)  #
            att = self.upsampler(a)
            attentions.append(att)
            gs.append(g_i)

        g = torch.cat(gs, dim=1)
        c = self.classify(g)

        return 0, c, attentions


class ResEncoderAtt(nn.Module):
    def __init__(self, inchannels, n_classes, first_channels=32, image_size=(256, 256, 256),
                 levels=5, dropout=False, dropout_rate=0.2, concatenation=False,
                 attention_layers_pos=(1, 2)):
        super(ResEncoderAtt, self).__init__()
        channels = first_channels
        self.N_CLASSES = n_classes
        self.levels = levels
        self.dropout = dropout
        self.kernel_size = 3
        self.stride = 2
        self.image_size = image_size
        self.concatenation = concatenation
        self.attention_layers_pos = attention_layers_pos
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
        encoders = [ResidualBlock(self.n_channels[0], self.n_channels[1], groupChannel=first_channels, dropout=self.dropout,
                                  dropout_rate=dropout_rate)]
        for i in range(1, self.levels):
            block = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ResidualBlock(self.n_channels[i], self.n_channels[i + 1], groupChannel=first_channels, dropout=self.dropout,
                              dropout_rate=dropout_rate)
            )
            encoders.append(block)
        self.encoders = nn.ModuleList(encoders)

        #
        #   Attention layers
        #
        self.attention_layers = nn.ModuleList(
            [MyLinearAttentionBlock(in_features=self.n_channels[-1], normalize_attn=True) for i in
             self.attention_layers_pos])

        projection = [
            nn.Conv3d(in_channels=self.n_channels[i + 1], out_channels=self.n_channels[-1], kernel_size=1, padding=0,
                      bias=False) for i in self.attention_layers_pos]
        self.projectors_layers = nn.ModuleList(
            projection
        )

        self.upsampler = nn.Upsample(size=tuple(self.image_size), mode='trilinear', align_corners=True)

        #
        #   Classification
        #
        self.dense = nn.Sequential(
            nn.Conv3d(self.n_channels[-1], self.n_channels[-1], kernel_size=self.layer_sizes[-1], stride=1, padding=0,
                      bias=True),
            nn.LayerNorm([self.n_channels[-1], 1, 1, 1]),
            nn.ReLU(inplace=True)
        )
        self.classify = LinearBlock(self.n_channels[-1] * len(self.attention_layers_pos), self.N_CLASSES)

    def forward(self, x):

        l_all = []
        l_i = x
        for i in range(self.levels):
            l_i = self.encoders[i](l_i)
            l_all.append(l_i)

        g = self.dense(l_i)

        attentions = []
        gs = []
        for i, layer_pos in enumerate(self.attention_layers_pos):
            p = self.projectors_layers[i](l_all[layer_pos])
            a, g_i = self.attention_layers[i](p, g)  #
            att = self.upsampler(a)
            attentions.append(att)
            gs.append(g_i)

        g = torch.cat(gs, dim=1)
        c = self.classify(g)

        return 0, c, attentions