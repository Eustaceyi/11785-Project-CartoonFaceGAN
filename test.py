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

model = ResnetGenerator(in_channels=3, out_channels=3, n_filters=4, n_blocks=1, use_dropout=False)

test_out1 = model(test_in1)
test_out2 = model(test_in2)

gan_criterion = GANLoss()
cycle_criterion = CycleLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
# loss1 = gan_criterion(test_out1, True)
# loss1.backward()
# optimizer.step()

loss2 = cycle_criterion(test_out1, test_out2)
loss2.backward()
optimizer.step()

print(test_out1.shape)