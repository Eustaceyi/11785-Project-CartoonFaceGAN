import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as utils
from model.loss import GANLoss, CycleLoss, IdentityLoss
from model.generator import ResnetGenerator, CycleGAN
from model.discriminator import Discriminator

# Hyper Parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1
num_epochs = 200
learning_rate = 1e-3

# Dataset
class Cycle_GAN_Dataset(datasets):
    def __init__(self, train_folder_A, train_folder_B):
        super(Cycle_GAN_Dataset, self).__init__()
        self.train_A = datasets.ImageFolder(train_folder_A, transform=transforms.ToTensor())
        self.train_B = datasets.ImageFolder(train_folder_B, transform=transforms.ToTensor())
        assert self.train_A.__len__() == self.train_B.__len__()

    def __getitem__(self, index):
        self.real_A = self.train_A.__getitem__(index)[0]
        self.real_B = self.train_B.__getitem__(index)[0]
        return self.real_A, self.real_B

    def __len__(self):
        return self.train_A.__len__()

train_path_A = 'C:/Users/eusta/Dropbox/Courses/11785/project/dataset/human_face/jundongy'
train_path_B = 'C:/Users/eusta/Dropbox/Courses/11785/project/dataset/emoji_face/jundongy'
train_dataset = Cycle_GAN_Dataset(train_path_A, train_path_B)

# Dataloader
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model


    
