import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as utils
from model.loss import GANLoss, CycleLoss, IdentityLoss
from model.generator import ResnetGenerator
from model.discriminator import Discriminator
from model.cyclegan import CycleGAN
from util.visualizer import Visualizer
from PIL import Image

# Hyper Parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1
num_epochs = 200
learning_rate = 1e-3

# Dataset
transform = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.ToTensor()])

class Cycle_GAN_Dataset(utils.Dataset):
    def __init__(self, train_folder_A, train_folder_B):
        super(Cycle_GAN_Dataset, self).__init__()
        self.train_A = datasets.ImageFolder(train_folder_A, transform=transform)
        self.train_B = datasets.ImageFolder(train_folder_B, transform=transform)
        assert self.train_A.__len__() == self.train_B.__len__()

    def __getitem__(self, index):
        self.real_A = self.train_A.__getitem__(index)[0]
        self.real_B = self.train_B.__getitem__(index)[0]
        return self.real_A, self.real_B

    def __len__(self):
        return self.train_A.__len__()

train_path_A = 'C:/Users/eusta/Dropbox/Courses/11785/project/dataset/human_face/'
train_path_B = 'C:/Users/eusta/Dropbox/Courses/11785/project/dataset/emoji_face/'
train_dataset = Cycle_GAN_Dataset(train_path_A, train_path_B)

# Dataloader
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
Model = CycleGAN(in_channels=3, out_channels=3, n_filters=64, n_blocks=6, n_sample=2)

# Visual
visual = Visualizer()

# Training Loop
total_step = len(train_loader)
print('Start Training!')
for epoch in range(num_epochs):
    Model.train()
    for i, (train_A, train_B) in enumerate(train_loader):
        train_A = train_A.to(device)
        train_B = train_B.to(device)

        fake_A, fake_B, recover_A, recover_B = Model.generator_forward(train_A, train_B)
        bce_A, bce_B = Model.discriminator_forward(fake_A, fake_B)
        Model.optim_params()
        # fake_A = transforms.ToPILImage()(fake_A.squeeze(0).detach().cpu()).convert('RGB')
        # fake_B = transforms.ToPILImage()(fake_B.squeeze(0).detach().cpu()).convert('RGB')
        # recover_A = transforms.ToPILImage()(recover_A.squeeze(0).detach().cpu()).convert('RGB')
        # recover_B = transforms.ToPILImage()(recover_B.squeeze(0).detach().cpu()).convert('RGB')
        # fake_A.save('fake_A.png')
        # fake_B.save('fake_B.png')
        # recover_A.save('recover_A.png')
        # recover_B.save('recover_B.png')

        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}]'
                    .format(epoch+1, num_epochs, i+1, total_step))
        if (i+1) % 50 == 0:    
            to_vis = [train_A.detach(), train_B.detach(), fake_A.detach(),
                    fake_B.detach(), recover_A.detach(), recover_B.detach()]
            to_vis = torch.stack(to_vis).squeeze(1)
            print(to_vis.shape)
            visual.plot_pictures(to_vis, epoch)
            # fake_A = transforms.ToPILImage()(fake_A.squeeze(0).detach().cpu()).convert('RGB')
            # fake_B = transforms.ToPILImage()(fake_B.squeeze(0).detach().cpu()).convert('RGB')
            # recover_A = transforms.ToPILImage()(recover_A.squeeze(0).detach().cpu()).convert('RGB')
            # recover_B = transforms.ToPILImage()(recover_B.squeeze(0).detach().cpu()).convert('RGB')
            # fake_A.save('fake_A_'+str(epoch)+'.png')
            # fake_B.save('fake_B_'+str(epoch)+'.png')
            # recover_A.save('recover_A_'+str(epoch)+'.png')
            # recover_B.save('recover_B_'+str(epoch)+'.png')







    
