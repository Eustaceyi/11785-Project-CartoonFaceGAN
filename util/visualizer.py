import numpy as np
import os
import sys
import ntpath   #for Windows user
import time
from util import utils
from subprocess import Popen, PIPE
from scipy.misc import imresize
import visdom

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    def __init__(self,display_id= 1, use_html= True, display_winsize= 256, display_name="test", display_port =8097, 
        display_ncols = 4, display_server ="http://localhost", display_env ='main', checkpoints_dir = './checkpoints',
        ):
        self.display_id = display_id
        self.use_html = use_html
        self.display_winsize = display_winsize
        self.display_port = display_port
        self.name = display_name
        self.saved = False
        self.display_server = display_server
        self.display_env = display_env

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            
            self.ncols = display_ncols
            self.vis = visdom.Visdom(server=self.display_server, port=self.display_port, env=self.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connection()

        #Create a checkpoints folder to save the intermediate progress
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(checkpoints_dir, self.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(checkpoints_dir, self.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def reset(self):
    	#reset the self.saved status
    	self.saved = False

    def create_visdom_connection(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)



    def plot_loss(self, epoch, progress_ratio, losses, legend = None):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)            -- current epoch
            progress_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
                                      = current total num of data being iterated / total datalength
            legend                  -- Determines how many different types of loss """
        #check if current object has the given attribute "plot data" or not
        if not hasattr(self, 'plot_data'):
            if legend != None:
                self.plot_data = {'X':[], 'Y':[], 'legend':[*legend]}
                try:
                    self.vis.line(X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                        Y=np.array(self.plot_data['Y']),
                        opts={
                        'title': self.name + ' loss over time',
                        'legend': self.plot_data['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},)
                except VisdomExceptionBase:
                    self.create_visdom_connection()
            else:
                self.plot_data = {'X':[], 'Y':[]}

        self.plot_data['X'].append(epoch + progress_ratio)
        #print('the current X is {}'.format(self.plot_data['X']))

        self.plot_data['Y'].append(losses)
        #print('the current loss is {}'.format(self.plot_data['Y']))
        try:
            self.vis.line(
                #X=np.stack([np.array(self.plot_data['X'])] * 1, 1),
                X=np.array(self.plot_data['X']),
                Y=np.array(self.plot_data['Y']),
                opts={
                'title': self.name + ' loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
                win = self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connection()


    def print_loss(self,epoch,iterations,loss,time_for_cal):
        """print current losses on console;
        Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losse       -- training losses stored in the format of (name, float) pairs
        t_comp      -- How long does it take to finish for this batch
        """
        on_screen = "epoch: {}, iter: {}, loss: {}, time_for_cal: {}".format(epoch, iterations, [*loss], time_for_cal)
        print(on_screen)








