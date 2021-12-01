#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-21 13:58
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/eriklindernoren/PyTorch-GAN

import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape, arch="mlp"):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.arch = arch
        self.model = None
        self.is_built = False
        self.build()

    def build(self):
        # MLP Discriminator
        if self.arch == "mlp":
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.input_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )

        self.is_built = True

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim, arch="mlp"):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.arch = arch
        self.model = None
        self.is_built = False
        self.build()

    def build(self):
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # MLP Discriminator
        if self.arch == "mlp":
            self.model = nn.Sequential(
                *block(self.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(self.input_shape))),
                nn.Tanh()
            )
        elif self.arch == "cnn":
            pass
        elif self.arch == "resnet":
            pass

        self.is_built = True

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.input_shape)
        return img
