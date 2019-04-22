# CartoonGAN

Use CycleGAN to transfer human face image into cartoon style. 


## Updates
### 04/04/2019 -- Enable visualizer
- New files:
	- visualizer_run.py: a simple CNN classification model, used to test visualizer
	- util/utils.py and util/visualizer.py: Enabled visdom for loss (single class loss)
	- Multi- loss should work, need to be tested
### 04/22/2019 
- New functions for Visualizer:
        - Enabled multiclass losses plot
	- Enabled plotting pictures on visdom server
