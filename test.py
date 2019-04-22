import torch
import numpy as np
import torch.optim as optim

from model.generator import *
from model.discriminator import *
from model.loss import *
from model.cyclegan import *

test_in1 = np.random.randn(2, 3, 128, 128)
test_in1 = torch.Tensor(test_in1)
test_in2 = np.random.randn(2, 3, 128, 128)
test_in2 = torch.Tensor(test_in2)
print(test_in1.shape)

cycle_gan = CycleGAN(in_channels=3, out_channels=3, n_filters=4, n_blocks=1)

gan_criterion = GANLoss()
cycle_criterion = CycleLoss()
identity_criterion = IdentityLoss()

cycle_optim = optim.Adam(cycle_gan.parameters(), lr=1e-3, weight_decay=5e-5)
d_optim = optim.Adam(cycle_gan.parameters(), lr=1e-3, weight_decay=5e-5)

fake_A, fake_B, recover_A, recover_B = cycle_gan.generator_forward(test_in1, test_in2)
bce_A, bce_B = cycle_gan.discriminator_forward(fake_A, fake_B)
gan_loss_a = gan_criterion(bce_A, is_real=False)
gan_loss_b = gan_criterion(bce_B, is_real=False)
cycle_loss_a = cycle_criterion(recover_A, test_in1)
cycle_loss_b = cycle_criterion(recover_B, test_in2)
identity_loss_a = identity_criterion(fake_A, test_in1)
identity_loss_b = identity_criterion(fake_B, test_in2)

# cycle_loss = cycle_loss_a + cycle_loss_b + identity_loss_b + identity_loss_b
# cycle_loss.backward()
# cycle_optim.step()

d_loss = gan_loss_a + gan_loss_b
d_loss.backward()
d_optim.step()

# print(cycle_loss.item())
print(d_loss.item())