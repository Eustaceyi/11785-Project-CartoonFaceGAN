'''This module contains functions that would be utilized for other modules'''
'''Consider that we use Python 3.6, we did not import "future" lib. For Python 2.7 user, uncomment the line below'''
# from future import __print_function 

import torch
import numpy as np
from PIL import Image
import os

def tensor2array(input_image, imtype=np.uint8):
	'''This function converts a tensor from Torch back to numpy array for reconstruct a picture

	Input parameters:

		tensor_image -- the input image array in the tensor form
		imtype       -- the desired type of converted numpy array
	'''
	if isinstance(input_image,np.ndarray):
		np_image = input_image

	else:
		if isinstance(input_image,torch.Tensor):
			tensor_image = input_image.data #extract the data out 
			np_image = tensor_image[0].cpu().float().numpy()  # convert it into a numpy array
			np_image = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # post-processing: tranpose and scaling

		else:
			return input_image

	return np_image.astype(imtype)

#-------------------------------------------------------------------------------------------------------------------#
def save_img(np_image,img_path):

	'''Reformat a numpy array into a picture and save it into the assigned path

	Input parameters:

		np_image -- Needs to be in numpy.ndarray
		img_path -- The path to be saved'''
	image_saved = Image.fromarray(np_image)
	image_saved.save(img_path)

#-------------------------------------------------------------------------------------------------------------------#
def mkdirs(paths):
    """create a new directory if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

#------------------------------------------------------------------------------------------------------------------#
def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy.astype(imtype)
    return image_numpy.transpose([2,0,1])


#TODO:
#def check_network(net, name = 'network'):
	# """Calculate and print the mean of average absolute(gradients)
    # Parameters:
    #     net (torch network) -- Torch network
    #     name (str) -- the name of the network
    # """


#TODO:
#def print_numpy(x, val= True, shp = False):
    # """Print the mean, min, max, median, std, and size of a numpy array
    # Parameters:
    #     val (bool) -- if print the values of the numpy array
    #     shp (bool) -- if print the shape of the numpy array
    # """
	

