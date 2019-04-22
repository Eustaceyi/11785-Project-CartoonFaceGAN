import torch
import numpy as np
import torch.optim as optim

from model.generator import *
from model.loss import *

test_in1 = np.random.randn(2, 3, 128, 128)
test_in1 = torch.Tensor(test_in1)
test_in2 = np.random.randn(2, 3, 128, 128)
test_in2 = torch.Tensor(test_in2)
print(test_in1.shape)

G = ResnetGenerator(in_channels=3, out_channels=3, n_filters=4, n_blocks=1, use_dropout=False)
F = ResnetGenerator(in_channels=3, out_channels=3, n_filters=4, n_blocks=1, use_dropout=False)

test_out1 = G(test_in1)
test_out2 = F(test_out1)

gan_criterion = GANLoss()
cycle_criterion = CycleLoss()
identity_criterion = IdentityLoss()

optimizer = optim.Adam(G.parameters(), lr=1e-3, weight_decay=5e-5)
gan_loss = gan_criterion(test_out1, is_real=False)
cycle_loss = cycle_criterion(test_out1, test_out2)
identity_loss = identity_criterion(test_out1, test_in1)
# loss1 = gan_criterion(test_out1, True)
# loss1.backward()
# optimizer.step()

loss = gan_loss + cycle_loss + identity_loss
loss.backward()
optimizer.step()

print(loss.item())
print(test_out1.shape)