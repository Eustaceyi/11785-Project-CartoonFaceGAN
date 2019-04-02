import numpy as np
import torch

import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.real_label = torch.Tensor(1.0)
        self.fake_label = torch.Tensor(0.0)

    def __get_target_tensor(self, predictions, is_real):
        target_tensor = self.real_label if is_real else self.fake_label
        return target_tensor.expand_as(predictions)
    
    def __call__(self, predictions, is_real):
        return self.forward(predictions, is_real)

    def forward(self, predictions, is_real):
        target_tensor = __get_target_tensor(predictions, is_real)
        return self.loss(predictions, target_tensor)


class CycleLoss(nn.L1Loss):
    def __init__(self, coef):
        super(CycleLoss, self).__init__()
        self.coef = coef
        # self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.forward(input, target)
    
    def forward(self, input, target):
        loss = super(CycleLoss, self).forward(input, target)
        return self.coef * loss
