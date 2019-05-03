import os
import torch
import argparse
import itertools
import torch.nn as nn
from generator import define_G
from discriminator import define_D
from util.image_pool import ImagePool
from skimage import color  # used for lab2rgb
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=3)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--netG', type=str, default='resnet_9block')
parser.add_argument('--netD', type=str, default='basic')
parser.add_argument('--norm', type=str, default='instance')
parser.add_argument('--no_dropout', type=bool, default=True)
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--init_gain', type=float, default=0.02)
parser.add_argument('--lambda_identity', type=float, default=0.5)
parser.add_argument('--lambda_A', type=float, default=10)
parser.add_argument('--lambda_B', type=float, default=10)
parser.add_argument('--pool_size', type=int, default=50)
parser.add_argument('--gan_mode', type=str, default='lsgan')
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--direction', type=str, default='AtoB')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--lr_policy', type=str, default='linear')
parser.add_argument('--dataroot', type=str,  required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
args = parser.parse_args() 
opt = args


#for test opts
parser_t = argparse.ArgumentParser(description='loader')
parser_t.add_argument('--dataroot', type=str,  required=True, , help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser_t.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser_t.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser_t.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser_t.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser_t.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser_t.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser_t.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser_t.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser_t.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser_t.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser_t.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser_t.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser_t.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
parser_t.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
parser_t.add_argument('--modelpath', type= str,  required=True, help = 'path to the model')
opt_test = parser_t.parse_args()






######################################################################################################################################################################################################################
input_nc = args.input_nc
output_nc = args.output_nc
ngf = args.ngf
ndf = args.ndf
netG = args.netG
netD = args.netD
norm = args.norm
no_dropout = args.no_dropout
init_type = args.init_type
init_gain = args.init_gain
lambda_identity = args.lambda_identity
lamb_B = args.lambda_B
lamb_A = args.lambda_A
pool_size = args.pool_size
gan_mode = args.gan_mode
lr_d = args.lr_d
lr_g = args.lr_g
beta1 = args.beta1
direction = args.direction
device = args.device
lr_policy = args.lr_policy

class CycleGANModel(nn.Module):
    def __init__(self, isTrain = True):
        super(CycleGANModel, self).__init__()
        self.netG_A = define_G(input_nc=input_nc, 
                                output_nc=output_nc, 
                                ngf=ngf, 
                                netG=netG, 
                                norm=norm,
                                no_dropout=no_dropout, 
                                init_type=init_type, 
                                init_gain=init_gain)

        self.netG_B = define_G(input_nc=input_nc, 
                                output_nc=output_nc, 
                                ngf=ngf, 
                                netG=netG, 
                                norm=norm,
                                no_dropout=no_dropout, 
                                init_type=init_type, 
                                init_gain=init_gain)
        
        if isTrain: #before was if self.isTrain
            self.netD_A = define_D(#output_nc=output_nc,
                                    input_nc = output_nc,
                                    ndf=ndf, 
                                    netD=netD,
                                    n_layers_D=3, 
                                    norm=norm, 
                                    init_type=init_type, 
                                    init_gain=init_gain)

            self.netD_B = define_D(#output_nc=output_nc,
                                    input_nc = output_nc,
                                    ndf=ndf, 
                                    netD=netD,
                                    n_layers_D=3, 
                                    norm=norm, 
                                    init_type=init_type, 
                                    init_gain=init_gain)
        
        if isTrain:
            if lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(input_nc == output_nc)
            self.fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss(gan_mode).to(device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr_g, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr_d, betas=(beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def return_resuts(self):
        return (self.fake_A, self.fake_B, self.rec_A, self.rec_B,
                self.loss_D_A, self.loss_D_B, self.loss_G_A, self.loss_G_B,
                self.loss_cycle_A, self.loss_cycle_B, self.loss_idt_A, self.loss_idt_B)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        AtoB = direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(device)
        self.real_B = input['B' if AtoB else 'A'].to(device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = lambda_identity
        lambda_A = lamb_A
        lambda_B = lamb_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def get_scheduler(self, optimizer):
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
                return lr_l
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def lab2rgb(self, L, AB):
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def compute_visuals(self):
        self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)
        self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss