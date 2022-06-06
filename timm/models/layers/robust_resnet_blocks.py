from functools import partial
import torch
import torch.nn as nn

from .drop import DropPath


class RobustResNetDWBlock(nn.Module):
    def __init__(self, indim, dim, drop_path=0., layer_scale_init_value=1e-6, mlp_ratio=4.,
                 norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                 stride=1, downsample=None,
                 kernel_size=11, padding=5):
        super().__init__()
        self.pwconv1 = nn.Conv2d(indim, dim, kernel_size=1, bias=True) # pointwise/1x1 convs, implemented with linear layers
        self.norm1 = norm_layer(dim)
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding,
                                 groups=dim, stride=stride, bias=True)  # depthwise conv
        self.pwconv2 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=1, bias=True)
        self.act3 = act_layer()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(int(mlp_ratio * dim))) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.conv_dw(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.drop_path(x) + shortcut
        x = self.act3(x)
        return x


class RobustResNetDWInvertedBlock(nn.Module):
    def __init__(self, indim, dim, drop_path=0., layer_scale_init_value=1e-6, mlp_ratio=4.,
                 norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                 stride=1, downsample=None,
                 kernel_size=11, padding=5):
        super().__init__()
        self.pwconv1 = nn.Conv2d(indim, int(mlp_ratio * dim), kernel_size=1, bias=True) # pointwise/1x1 convs, implemented with linear layers
        self.norm1 = norm_layer(int(mlp_ratio * dim))
        self.act1 = act_layer()
        self.conv_dw = nn.Conv2d(int(mlp_ratio * dim), int(mlp_ratio * dim), kernel_size=kernel_size, padding=padding,
                                 groups=int(mlp_ratio * dim), stride=stride, bias=True)  # depthwise conv
        self.pwconv2 = nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.drop_path(x) + shortcut
        return x


class RobustResNetDWUpInvertedBlock(nn.Module):
    def __init__(self, indim, dim, drop_path=0., layer_scale_init_value=1e-6, mlp_ratio=4.,
                 norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                 stride=1, downsample=None,
                 kernel_size=11, padding=5):
        super().__init__()
        self.conv_dw = nn.Conv2d(indim, dim, kernel_size=kernel_size, padding=padding, groups=indim, stride=stride, bias=True)  # depthwise conv
        self.norm1 = norm_layer(dim)
        self.pwconv1 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=1, bias=True)
        self.act2 = act_layer()
        self.pwconv2 = nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.drop_path(x) + shortcut
        return x


class RobustResNetDWDownInvertedBlock(nn.Module):
    def __init__(self, indim, dim, drop_path=0., layer_scale_init_value=1e-6, mlp_ratio=4.,
                 norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, partial=True),
                 stride=1, downsample=None,
                 kernel_size=11, padding=5):
        super().__init__()
        self.pwconv1 = nn.Conv2d(indim, int(mlp_ratio * dim), kernel_size=1, bias=True)
        self.norm2 = norm_layer(int(mlp_ratio * dim))
        self.act2 = act_layer()
        self.pwconv2 = nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1, bias=True)
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, stride=stride, bias=True)  # depthwise conv

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.pwconv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = self.conv_dw(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.drop_path(x) + shortcut
        return x