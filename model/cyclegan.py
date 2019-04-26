import torch
import numpy as np
import torch.nn as nn
import itertools
# import discriminator
from model.discriminator import *
from model.generator import *
from model.loss import *

learning_rate = 1e-3

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, n_blocks, n_sample, use_dropout=False, use_bias=False):
        super(CycleGAN, self).__init__()
        self.G = ResnetGenerator(in_channels, out_channels, n_filters, n_blocks, n_sample, use_dropout).cuda()
        self.F = ResnetGenerator(in_channels, out_channels, n_filters, n_blocks, n_sample, use_dropout).cuda()
        self.D_x = Discriminator().cuda()
        self.D_y = Discriminator().cuda()
        self.GANLoss = GANLoss()
        self.CycleLoss = CycleLoss()
        self.IdtLoss = IdentityLoss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=learning_rate)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_x.parameters(), self.D_y.parameters()), lr=learning_rate)
        

    def generator_forward(self, input_A, input_B):
        self.real_A = input_A
        self.real_B = input_B
        self.fake_B = self.G(self.real_A) # fB = G_A(A)
        self.recover_A = self.F(self.fake_B) # rA = G_B(G_A(A))
        self.fake_A = self.F(self.real_B) # fA = G_B(B)
        self.recover_B = self.G(self.fake_A) # rB = G_A(G_B(B))

        return self.fake_A, self.fake_B, self.recover_A, self.recover_B

    def discriminator_forward(self, input_A, input_B):
        return self.D_x(input_A), self.D_y(input_B)

    def discriminator_backward_basic(self, discriminator, real, fake):
        pred_real = discriminator(real)
        D_loss_real = self.GANLoss(pred_real, True)

        pred_fake = discriminator(fake.detach())
        D_loss_fake = self.GANLoss(pred_fake, False)

        D_loss = (D_loss_fake + D_loss_real) * 0.5
        D_loss.backward()
        return D_loss

    def D_x_backward(self):
        self.D_x_loss = self.discriminator_backward_basic(self.D_x, self.real_B, self.fake_B)

    def D_y_backward(self):
        self.D_y_loss = self.discriminator_backward_basic(self.D_y, self.real_A, self.fake_A)

    def G_backward(self):
        self.idt_A = self.G(self.real_B)
        self.idt_B = self.F(self.real_A)
        self.idt_loss_A = self.IdtLoss(self.idt_A, self.real_B) * 5
        self.idt_loss_B = self.IdtLoss(self.idt_B, self.real_A) * 5
        self.G_loss = self.GANLoss(self.D_x(self.fake_B), True)
        self.F_loss = self.GANLoss(self.D_y(self.fake_A), True)
        self.cycleA_loss = self.CycleLoss(self.recover_A, self.real_A) * 10
        self.cycleB_loss = self.CycleLoss(self.recover_B, self.real_B) * 10
        self.generator_loss = self.G_loss + self.F_loss + self.cycleA_loss + self.cycleB_loss + self.idt_loss_A + self.idt_loss_B
        self.generator_loss.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optim_params(self):
        self.set_requires_grad([self.D_x, self.D_y], False)
        self.optimizer_G.zero_grad()
        self.G_backward()
        self.optimizer_G.step()

        self.set_requires_grad([self.D_x, self.D_y], True)
        self.optimizer_D.zero_grad()
        self.D_x_backward()
        self.D_y_backward()
        self.optimizer_D.step()