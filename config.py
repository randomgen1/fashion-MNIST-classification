'''
This file is used to choose model architecture and model parameters
'''
import torch
import os

################################################################################
experiment_name = 'VGG_like_CNN_1_model'

# Model parameters
model = 'VGG_like_CNN' # 'LeNet_like_CNN' # 'resnet10' # 'resnet18_pretrained'
learning_rate = 0.002
batch_size = 128
epochs = 2
n_classes = 10

criterion = torch.nn.CrossEntropyLoss() # loss function

cwd = os.getcwd()
logdir = os.path.join(cwd, str(experiment_name)) # path where model will be saved
logdir_visualise = os.path.join(logdir, 'logs') # for tensorboard

if not os.path.exists(logdir_visualise):
    os.makedirs(logdir_visualise)

################################################################################
