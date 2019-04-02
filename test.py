import torch
import numpy as np
from model.generator import *

test_in = np.random.randn(32, 3, 128, 128)
test_in = torch.Tensor(test_in)
print(test_in.shape)

model = ResnetGenerator(in_channels=3, out_channels=3, n_filters=64, n_blocks=9, use_dropout=False)

test_out = model(test_in)
print(test_out.shape)