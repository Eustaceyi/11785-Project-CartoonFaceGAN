import numpy as np
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).cuda()

    def __call__(self, prediction, target_is_real):
        return self.forward(prediction, target_is_real)
    
    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


# class GANLoss(nn.Module):
#     def __init__(self):
#         super(GANLoss, self).__init__()
#         self.loss = nn.BCEWithLogitsLoss()
#         self.real_label = torch.tensor(1.0)
#         self.fake_label = torch.tensor(0.0)

#     def __get_target_tensor(self, predictions, is_real):
#         target_tensor = self.real_label if is_real else self.fake_label
#         return target_tensor.expand_as(predictions).cuda()
    
#     def __call__(self, predictions, is_real):
#         return self.forward(predictions, is_real)

#     def forward(self, predictions, is_real):
#         target_tensor = self.__get_target_tensor(predictions, is_real)
#         return self.loss(predictions, target_tensor)


class CycleLoss(nn.L1Loss):
    def __init__(self, coef=10.0):
        super(CycleLoss, self).__init__()
        self.coef = coef

    def __call__(self, input, target):
        return self.forward(input, target)
    
    def forward(self, input, target):
        loss = super(CycleLoss, self).forward(input, target)
        return self.coef * loss

    def update_coef(self, coef):
        self.coef = coef


class IdentityLoss(nn.L1Loss):
    def __init__(self, coef=10.0):
        super(IdentityLoss, self).__init__()
        self.coef = coef

    def __call__(self, input, target):
        return self.forward(input, target)
    
    def forward(self, input, target):
        loss = super(IdentityLoss, self).forward(input, target)
        return self.coef * loss

    def update_coef(self, coef):
        self.coef = coef
