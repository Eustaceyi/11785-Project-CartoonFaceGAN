import torch
import numpy as np
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        
        block = []
        block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), 
            # nn.BatchNorm2d(dim),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            block +=[nn.Dropout(0.5)]

        block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            # nn.BatchNorm2d(dim)
            norm_layer(dim)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, 
        in_channels=3, 
        out_channels=3, 
        n_filters=64, 
        n_blocks=6, 
        n_sample=2, 
        use_dropout=False, 
        norm_layer_type='instance'
    ):
        super(ResnetGenerator, self).__init__()
        if norm_layer_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('{} not implemented'.format(norm_layer_type))
        
        model = [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(
                in_channels, 
                n_filters, 
                kernel_size=7, 
                padding=0, 
                bias=False),
            # nn.BatchNorm2d(n_filters),
            norm_layer(n_filters),
            nn.ReLU(inplace=True)
        ]

        n_downsample = n_sample
        n_upsample = n_sample

        # downsampling layers
        # downsample to N/(2^layers) X N/(2^layers)
        for i in range(n_downsample):
            cur_n_filters = n_filters * 2**i
            model += [
                nn.Conv2d(
                    cur_n_filters, 
                    cur_n_filters*2, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    bias=False),
                # nn.BatchNorm2d(cur_n_filters*2),
                norm_layer(cur_n_filters * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Resnet blocks
        for i in range(n_blocks):
            model += [
                ResnetBlock(cur_n_filters*2, norm_layer, use_dropout=False, use_bias=False)
            ]
        
        # upsampling layers
        for i in range(n_upsample):
            cur_n_filters = n_filters * 2**(n_upsample - i)
            model += [
                nn.ConvTranspose2d(
                    cur_n_filters, 
                    cur_n_filters//2, 
                    kernel_size=3, 
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                    ), 
                # nn.BatchNorm2d(cur_n_filters//2),
                norm_layer(cur_n_filters // 2),
                nn.ReLU(inplace=True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

    def weights_init(self):
        pass
