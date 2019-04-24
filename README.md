# CartoonGAN

Use CycleGAN to transfer human face image into cartoon style. 


## Updates
### 03/31/2019
- New files:
	- generator.py: generator with resnet block
	- discriminator: binary classifier with CNN
	- loss.py: GAN loss, cycle loss and identity loss
	- options.py: arguments
### 04/04/2019 -- Enable visualizer
- New files:
	- visualizer_run.py: a simple CNN classification model, used to test visualizer
	- util/utils.py and util/visualizer.py: Enabled visdom for loss (single class loss)
	- Multi- loss should work, need to be tested
### 04/22/2019 
- New functions:
	- Enabled multi-class losses plotting
	- Enabled plotting pictures onto visdom
- New files:
	- cyclegan.py: wrap the generators and discriminators
	- train.py: train the model
	- code_test.py: using for debug the code
