import time
import torch
from util.visualizer import Visualizer
from util.utils import save_img
from util.dataloader import *
from CycleGAN import *
from util.utils import tensor2im

if __name__ == '__main__':   # get training options
    dataset = loader_main(opt)        # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    model = CycleGANModel()
    visual= Visualizer()   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    results = None

    for epoch in range(300):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        fake_loss = 1 
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print(data)
            epoch_iter +=1
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += 1
            epoch_iter += 1

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % 1 == 0:   # display images on visdom and save images to a HTML file
                results = model.return_resuts() 

                fake_loss +=1
                losses = [results[4].cpu().item(),
                            results[5].cpu().item(),
                            results[6].cpu().item(),
                            results[7].cpu().item(),
                            results[8].cpu().item(),
                            results[9].cpu().item(), 
                            results[10].cpu().item(), 
                            results[11].cpu().item()]
                    
                legend = ["D_A_loss", 
                            "D_B_loss", 
                            "G_A_loss",    
                            "G_B_loss", 
                            "Cycle_A_Loss", 
                            "Cycle_B_loss",
                            "IdentityA_loss",
                            "IdentityB_loss"]

                visual.plot_loss(epoch, epoch_iter / len(dataset), losses, legend)

                fake_A, fake_B, recover_A, recover_B = results[0], results[1],results[2],results[3]
                to_vis = [tensor2im(data['A'].cpu().detach()), 
                            tensor2im(data['B'].cpu().detach()),
                            tensor2im(fake_A.cpu().detach()), 
                            tensor2im(fake_B.cpu().detach()), 
                            tensor2im(recover_A.cpu().detach()), 
                            tensor2im(recover_B.cpu().detach())]

                visual.plot_pictures(to_vis, epoch)
                
        torch.save(model.state_dict(), 'model_' + str(epoch) +'.ckpt')
        save_img(to_vis[0].astype('uint8').transpose(1,2,0), './checkpoints/real_A' + str(epoch) + '.png')
        save_img(to_vis[1].astype('uint8').transpose(1,2,0), './checkpoints/real_B' + str(epoch) + '.png')
        save_img(to_vis[2].astype('uint8').transpose(1,2,0), './checkpoints/fake_A' + str(epoch) + '.png')
        save_img(to_vis[3].astype('uint8').transpose(1,2,0), './checkpoints/fake_B' + str(epoch) + '.png')
        save_img(to_vis[4].astype('uint8').transpose(1,2,0), './checkpoints/rec_B' + str(epoch) + '.png')
        save_img(to_vis[5].astype('uint8').transpose(1,2,0), './checkpoints/rec_B' + str(epoch) + '.png')
        