# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SelectAdaptivePool2d, AvgPool2dSame
from .layers import RobustResNetDWBlock, RobustResNetDWInvertedBlock, RobustResNetDWUpInvertedBlock, RobustResNetDWDownInvertedBlock
from .registry import register_model


__all__ = ['RobustResNet']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    small=_cfg(),
    base=_cfg(),
)


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=True),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True),
        norm_layer(out_channels)
    ])


class Stage(nn.Module):
    def __init__(
            self, block_fn, in_chs, chs, stride=2, depth=2, dp_rates=None, layer_scale_init_value=1.0,
            norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, partial=True),
            avg_down=False, down_kernel_size=1, mlp_ratio=4., inverted=False, **kwargs):
        super().__init__()

        blocks = []
        dp_rates = dp_rates or [0.] * depth
        for block_idx in range(depth):
            stride_block_idx = depth - 1 if block_fn == RobustResNetDWDownInvertedBlock else 0
            current_stride = stride if block_idx == stride_block_idx else 1

            downsample = None
            if inverted:
                if in_chs != chs or current_stride > 1:
                    down_kwargs = dict(
                        in_channels=in_chs, out_channels=chs, kernel_size=down_kernel_size,
                        stride=current_stride, norm_layer=norm_layer)
                    downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
            else:
                if in_chs != int(mlp_ratio * chs) or current_stride > 1:
                    down_kwargs = dict(
                        in_channels=in_chs, out_channels=int(mlp_ratio * chs), kernel_size=down_kernel_size,
                        stride=current_stride, norm_layer=norm_layer)
                    downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
            if downsample is not None:
                assert block_idx in [0, depth - 1]

            blocks.append(block_fn(
                indim=in_chs, dim=chs, drop_path=dp_rates[block_idx], layer_scale_init_value=layer_scale_init_value,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer, act_layer=act_layer,
                stride=current_stride,
                downsample=downsample,
                **kwargs,
            ))
            in_chs = int(chs * mlp_ratio) if not inverted else chs

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class RobustResNet(nn.Module):
    # TODO: finish comment here
    r""" RobustResNetDW
        A PyTorch impl of :

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self, block_fn, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32,
            patch_size=16, stride_stage=(3, ),
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),  layer_scale_init_value=1e-6,
            head_init_scale=1., head_norm_first=False,
            norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
            drop_rate=0., drop_path_rate=0., mlp_ratio=4., block_args=None,
    ):
        super().__init__()
        assert block_fn in [RobustResNetDWBlock, RobustResNetDWInvertedBlock, RobustResNetDWUpInvertedBlock, RobustResNetDWDownInvertedBlock]
        self.inverted = True if block_fn != RobustResNetDWBlock else False
        assert output_stride == 32

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        block_args = block_args or dict()
        print(f'using block args: {block_args}')

        assert patch_size == 16
        self.stem = nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size)
        curr_stride = patch_size

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_chs = dims[0]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if i in stride_stage else 1
            curr_stride *= stride
            chs = dims[i]
            stages.append(Stage(
                block_fn, prev_chs, chs, stride=stride,
                depth=depths[i], dp_rates=dp_rates[i], layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer, act_layer=act_layer, mlp_ratio=mlp_ratio,
                inverted=self.inverted, **block_args)
            )
            prev_chs = int(mlp_ratio * chs) if not self.inverted else chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        assert curr_stride == output_stride

        self.num_features = prev_chs

        self.norm_pre = nn.Identity()
        self.head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
            # ('norm', norm_layer(self.num_features)),
            ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        self.resnet_init_weights()

    def resnet_init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
        # pool -> norm -> fc
        self.head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
            ('norm', self.head.norm),
            ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_robust_resnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        RobustResNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


@register_model
def robust_resnet_dw_small(pretrained=False, **kwargs):
    '''
    4.49GFLOPs and 38.6MParams
    '''
    assert not pretrained, 'no pretrained models!'
    model_args = dict(block_fn=RobustResNetDWBlock, depths=(3, 4, 12, 3), dims=(96, 192, 384, 768),
                      block_args=dict(kernel_size=11, padding=5),
                      patch_size=16, stride_stage=(3,),
                      norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                      **kwargs)
    model = _create_robust_resnet('small', pretrained=pretrained, **model_args)
    return model


@register_model
def robust_resnet_inverted_dw_small(pretrained=False, **kwargs):
    '''
    4.59GFLOPs and 33.6MParams
    '''
    assert not pretrained, 'no pretrained models!'
    model_args = dict(block_fn=RobustResNetDWInvertedBlock, depths=(3, 4, 14, 3), dims=(96, 192, 384, 768),
                      block_args=dict(kernel_size=7, padding=3),
                      patch_size=16, stride_stage=(3,),
                      norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                      **kwargs)
    model = _create_robust_resnet('small', pretrained=pretrained, **model_args)
    return model


@register_model
def robust_resnet_up_inverted_dw_small(pretrained=False, **kwargs):
    '''
    4.43GFLOPs and 34.4MParams
    '''
    assert not pretrained, 'no pretrained models!'
    model_args = dict(block_fn=RobustResNetDWUpInvertedBlock, depths=(3, 4, 14, 3), dims=(96, 192, 384, 768),
                      block_args=dict(kernel_size=11, padding=5),
                      patch_size=16, stride_stage=(3,),
                      norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                      **kwargs)
    model = _create_robust_resnet('small', pretrained=pretrained, **model_args)
    return model


@register_model
def robust_resnet_down_inverted_dw_small(pretrained=False, **kwargs):
    '''
    4.55GFLOPs and 24.3MParams
    '''
    assert not pretrained, 'no pretrained models!'
    model_args = dict(block_fn=RobustResNetDWDownInvertedBlock, depths=(3, 4, 15, 3), dims=(96, 192, 384, 768),
                      block_args=dict(kernel_size=11, padding=5),
                      patch_size=16, stride_stage=(2,),
                      norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                      **kwargs)
    model = _create_robust_resnet('small', pretrained=pretrained, **model_args)
    return model