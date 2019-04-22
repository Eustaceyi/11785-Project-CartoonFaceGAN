import torch
import numpy as np
import torch.nn as nn
# import discriminator
from model.discriminator import *
from model.generator import *

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, n_blocks, use_dropout=False, use_bias=False):
        super(CycleGAN, self).__init__()
        self.G = ResnetGenerator(in_channels, out_channels, n_filters, n_blocks, use_dropout)
        self.F = ResnetGenerator(in_channels, out_channels, n_filters, n_blocks, use_dropout)
        self.D_x = Discriminator()
        self.D_y = Discriminator()
        

    def generator_forward(self, input_A, input_B):
        real_A = input_A
        real_B = input_B
        fake_B = self.G(real_A) # fB = G_A(A)
        recover_A = self.F(fake_B) # rA = G_B(G_A(A))
        fake_A = self.F(real_B) # fA = G_B(B)
        recover_B = self.G(fake_A) # rB = G_A(G_B(B))

        return fake_A, fake_B, recover_A, recover_B

    def discriminator_forward(self, input_A, input_B):
        return self.D_x(input_A), self.D_y(input_B)


