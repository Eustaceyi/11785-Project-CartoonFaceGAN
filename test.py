import time
import torch
from util.visualizer import Visualizer
from util.utils import save_img
from util.dataloader import *
from CycleGAN import *
from util.utils import tensor2im

if __name__ == '__main__':
    dataset = loader_main(opt)

    model = CycleGANModel()
    model.load_state_dict(torch.load('C:/Users/eusta/Dropbox/Courses/11785/project/model_295.ckpt'))
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            fake_A, fake_B = model.forward_test()
            fake_A = tensor2im(fake_A.cpu().detach())
            fake_B = tensor2im(fake_B.cpu().detach())
            save_img(fake_A.astype('uint8').transpose(1,2,0), './checkpoints/fake_A_' + str(i) + '.png')
            save_img(fake_B.astype('uint8').transpose(1,2,0), './checkpoints/fake_B_' + str(i) + '.png')