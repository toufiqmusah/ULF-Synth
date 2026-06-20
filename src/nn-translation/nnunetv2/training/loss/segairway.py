import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from torch.nn import Parameter as P

from functools import partial

from typing import Optional
from torch import Tensor
from typing import List, Type, Union

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, stride, dilation=1):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride, dilation=dilation)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, stride=1, dilation=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, stride=stride, dilation=dilation):
            self.add_module(name, module)

class AttModule(nn.Module):
    def __init__(self, channel, mid_channel=8):
        super(AttModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
            
class Encoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, apply_pooling=True, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order='gcr', num_groups=8, padding=1, stride=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        self.conv1 = SingleConv(in_channels, middle_channels, conv_kernel_size, conv_layer_order, num_groups, padding=padding, stride=stride)
        self.conv2 = SingleConv(middle_channels, out_channels, conv_kernel_size, conv_layer_order, num_groups, padding=padding, stride=stride)
        self.dilation_conv = SingleConv(middle_channels, middle_channels, (3,3,3), conv_layer_order, num_groups, padding=(4,4,4), stride=stride, dilation=4)
        self.att = AttModule(channel=middle_channels)
        
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.conv1(x)

        x = x+self.att(self.dilation_conv(x))
        x = self.conv2(x)        
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, upsample_out_channels, conv_in_channels, conv_middle_channels, out_channels, conv_kernel_size=3, conv_layer_order='gcr', num_groups=8, conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, upsample_out_channels, kernel_size=deconv_kernel_size, stride=deconv_stride, padding=deconv_padding)
        self.joining = partial(self._joining, concat=True)
        self.conv1 = SingleConv(conv_in_channels, conv_middle_channels, conv_kernel_size, conv_layer_order, num_groups, padding=conv_padding, stride=conv_stride)
        self.conv2 = SingleConv(conv_middle_channels, out_channels, conv_kernel_size, conv_layer_order, num_groups, padding=conv_padding, stride=conv_stride)
        self.dilation_conv = SingleConv(conv_middle_channels, conv_middle_channels, (3,3,3), conv_layer_order, num_groups, padding=(4,4,4), stride=conv_stride, dilation=4)
        self.att = AttModule(channel=conv_middle_channels)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = self.joining(encoder_features, x)
        x = self.conv1(x)
        x = x+self.att(self.dilation_conv(x))
        x = self.conv2(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x
            
class SegAirwayModel(nn.Module):
    def __init__(self, in_channels, out_channels, layer_order='gcr', **kwargs):
        super(SegAirwayModel, self).__init__()
        encoder_1 = Encoder(in_channels=in_channels, middle_channels=16, out_channels=32, apply_pooling=False, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1, stride=1)
        encoder_2 = Encoder(in_channels=32, middle_channels=32, out_channels=64, apply_pooling=True, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1, stride=1)
        encoder_3 = Encoder(in_channels=64, middle_channels=64, out_channels=128, apply_pooling=True, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1, stride=1)
        encoder_4 = Encoder(in_channels=128, middle_channels=128, out_channels=256, apply_pooling=True, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1, stride=1)
        self.encoders = nn.ModuleList([encoder_1, encoder_2, encoder_3, encoder_4])

        decoder_1 = Decoder(in_channels=256, upsample_out_channels=256, conv_in_channels=384, conv_middle_channels=128, out_channels=128, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=8, conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1)
        decoder_2 = Decoder(in_channels=128, upsample_out_channels=128, conv_in_channels=192, conv_middle_channels=64, out_channels=64, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=8, conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1)
        decoder_3 = Decoder(in_channels=64, upsample_out_channels=64, conv_in_channels=96, conv_middle_channels=32, out_channels=32, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=8, conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1)
        self.decoders = nn.ModuleList([decoder_1, decoder_2, decoder_3])

        self.final_conv = nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        encoders_features, decoders_features = [], []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        for decoder, encoder_features in zip(self.decoders, encoders_features[1:]):
            x = decoder(encoder_features, x)
            decoders_features.append(x)
        final_conv = self.final_conv(x)
        x = self.final_activation(final_conv)
        return encoders_features[::-1] + decoders_features + [final_conv] + [x]