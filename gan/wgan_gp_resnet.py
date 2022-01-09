#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-21 18:40
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py
# @RefLink : https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
CONDITIONAL = True  # Whether to train a conditional or unconditional model
ACGAN = True  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (32*32*3)


def space_to_depth(x, block_size):
    N, C, H, W = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(N, C * block_size ** 2, H // block_size, W // block_size)


class UpSample(nn.Module):
    """UpSample
    TODO
    """
    pass


def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """Conditional batchnorm for BCHW conv filtermaps
    # TODO 改成 PyTorch
    References:
    [1] https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py
    [2] Conditional batchnorm (dumoulin et al 2016)
    """
    if axes != [0, 2, 3]:
        raise Exception('unsupported')

    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list()  # shape is [1,n,1,1]
    offset_m = lib.param(
        name+'.offset', np.zeros([n_labels, shape[1]], dtype='float32'))
    scale_m = lib.param(
        name+'.scale', np.ones([n_labels, shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(
        inputs, mean, var, offset[:, :, None, None], scale[:, :, None, None], 1e-5)

    return result


class ResidualBlock(nn.Module):
    """
    resample: None, 'down', or 'up'
    """

    def __init__(self, input_dim, output_dim, kernel_size, resample=None, no_dropout=False, labels=None, normalize_type=None):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.no_dropout = no_dropout
        self.labels = labels
        self.normalize_type = normalize_type

        if resample == 'down':
            self.conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size)
            # ConvMeanPool
            self.conv_2 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size),
                nn.AvgPool2d(2)
            )
            # ConvMeanPool
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1),
                nn.AvgPool2d(2)
            )
        elif resample == 'up':
            # UpsampleConv
            self.conv_1 = nn.Sequential(
                UpSample(2),
                nn.Conv2d(input_dim, output_dim, kernel_size)
            )
            # UpsampleConv
            self.conv_shortcut = nn.Sequential(
                UpSample(2),
                nn.Conv2d(input_dim, output_dim, kernel_size=1)
            )
            self.conv_2 = nn.Conv2d(input_dim, output_dim, kernel_size)
        elif resample == None:
            self.conv_shortcut = nn.Conv2d(
                input_dim, output_dim, kernel_size=1)
            self.conv_1 = nn.Conv2d(input_dim, output_dim, kernel_size)
            self.conv_2 = nn.Conv2d(input_dim, output_dim, kernel_size)
        else:
            raise Exception('invalid resample value')
        # normalize_type
        # Discriminator Layernorm
        # Generator Batchnorm
        if normalize_type == "layernorm":
            self.norm = Layernorm()
        elif normalize_type == "batchnorm":
            self.norm = Batchnorm()
        else:
            raise ValueError()

    def forword(self, inputs, labels=None):
        if self.output_dim == self.input_dim and self.resample == None:
            shortcut = inputs  # Identity skip-connection
        else:
            shortcut = self.conv_shortcut(inputs)

        output = inputs
        output = self.norm(output, labels=labels)
        output = F.relu(output)
        output = self.conv_1(inputs=output)
        output = self.norm(output, labels=labels)
        output = F.relu(output)

        output = self.conv_2(output)
        output = self.pool(output)

        return shortcut + output


class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        super(Generator, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.residual_block_1 = ResidualBlock(
            latent_dim, latent_dim, 3,  resample='up')
        self.residual_block_2 = ResidualBlock(
            latent_dim, latent_dim, 3,  resample='up')
        self.residual_block_3 = ResidualBlock(
            latent_dim, latent_dim, 3,  resample='up')

        self.fc1 = nn.Linear(latent_dim, 4*4*latent_dim)

        self.output_conv = nn.Conv2d(latent_dim, 3, 3)

    def forward(self, n_samples, labels, noise=None):
        dim_g = self.latent_dim
        if noise is None:
            noise = torch.randn(n_samples, dim_g)

        output = self.fc1(noise)
        output = output.reshape(output, [-1, dim_g, 4, 4])

        output = self.residual_block_1(output, labels)
        output = self.residual_block_2(output, labels)
        output = self.residual_block_3(output, labels)

        output = Normalize('Generator.OutputN', output)
        output = F.relu(output)

        output = self.output_conv(output)
        output = torch.tanh(output)

        return torch.reshape(output, [-1, OUTPUT_DIM])


class OptimizedResBlockDisc1(nn.modules):
    def __init__(self):

        self.conv_1 = nn.Conv2d(3, DIM_D, kernel_size=3)
        # MeanPoolConv
        self.conv_2 = nn.Sequential(
            nn.Conv2d(DIM_D, DIM_D, kernel_size=3),
            nn.AvgPool2d(2)
        )
        self.conv_shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(3, DIM_D, kernel_size=1)
        )

    def forward(self, inputs):
        shortcut = self.conv_shortcut(inputs)
        output = inputs
        output = self.conv_1(output)
        output = F.relu(output)
        output = self.conv_2(output)

        return shortcut + output


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape

        self.nonlinearity = nn.ReLU()

        self.fc = nn.Linear(DIM_D, 1)
        self.fc_acgan = nn.Linear(DIM_D, 10)

    def forward(self, inputs, labels):
        output = inputs
        output = output.reshape(output, [-1, 3, 32, 32])
        output = OptimizedResBlockDisc1(output)

        output = ResidualBlock('Discriminator.2', DIM_D, DIM_D,
                               3, output, resample='down', labels=labels)
        output = ResidualBlock('Discriminator.3', DIM_D, DIM_D,
                               3, output, resample=None, labels=labels)
        output = ResidualBlock('Discriminator.4', DIM_D, DIM_D,
                               3, output, resample=None, labels=labels)

        output = self.nonlinearity(output)
        output = torch.mean(output, dim=[2, 3])
        output_wgan = self.fc(output)
        output_wgan = torch.flatten(output_wgan, [-1])

        output_acgan = None
        if CONDITIONAL and ACGAN:
            output_acgan = self.fc_acgan(output)

        return output_wgan, output_acgan
