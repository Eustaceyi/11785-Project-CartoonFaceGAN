import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_pic_chan=3, num_layers=3, norm_layer_type='batch'):
        '''
        Params:
            input_pic_chan (int, default=3): number of channels in input image
            num_layers (int, default=3): number of middle layers in discriminator
            norm_layer (string, default='batch'): type of norm layer, options: 'batch'/'instance'
        '''
        super(Discriminator, self).__init__()
        if norm_layer_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('{} not implemented'.format(norm_layer_type))
        
        if num_layers <=3:
            sequence = [nn.Conv2d(input_pic_chan, 64, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, True)]
            hid_chan = 64
            for i in range(1,num_layers+1):
                prev_hid_chan = hid_chan
                hid_chan *= 2
                sequence += [nn.Conv2d(prev_hid_chan, hid_chan, kernel_size=4, stride=2, padding=1, bias=False),
                            norm_layer(hid_chan),
                            nn.LeakyReLU(0.2, True)]
            sequence += [nn.Conv2d(hid_chan, 1, kernel_size=4, stride=1, padding=1)]
        else:
            sequence = [nn.Conv2d(input_pic_chan, 64, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                        norm_layer(128),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                        norm_layer(256),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                        norm_layer(512),
                        nn.LeakyReLU(0.2, True)]
            for i in range(3,num_layers):
                sequence += [nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
                            norm_layer(512),
                            nn.LeakyReLU(0.2, True)]
            sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)