#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import torch
from torch import nn
import my_nntools as nt
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

import itertools

import models as models
from utils import *


# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[5]:


dataset_root_dir="./datasets/apple2orange/"


# In[42]:


# Dataset loader
train_set = ImageDataset(dataset_root_dir, image_size=256, unaligned=True, mode='train')
val_set = ImageDataset(dataset_root_dir, image_size=256, unaligned=True, mode='val')
test_set = ImageDataset(dataset_root_dir, image_size=256, unaligned=True, mode='test')


# In[2]:


class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.MSE_Loss = nn.MSELoss()
        self.L1_Loss = nn.L1Loss()
    
    def criterion(self, y, d):
        return self.cross_entropy(y, d)    
    def criterion_GAN(self, y, d):
        return self.MSE_Loss(y, d)
    def criterion_cycle(self, y, d):
        return self.L1_Loss(y, d)*5
    def criterion_identity(self, y, d):
        return self.L1_Loss(y, d)*10
    
class C_GAN(NNClassifier):
    def __init__(self, fine_tuning=True):
        super(C_GAN, self).__init__()
        
        self.G_A2B = models.Generator(3, 3)
        self.G_B2A = models.Generator(3, 3)
        self.D_A = models.Discriminator(3)
        self.D_B = models.Discriminator(3)
        
        self.G_A2B.apply(init_parameters)
        self.G_A2B.apply(init_parameters)
        self.D_A.apply(init_parameters)
        self.D_B.apply(init_parameters)
    
        self.fake_a_buffer = ReplayBuffer()
        self.fake_b_buffer = ReplayBuffer()
    
    def forward(self, real_a, real_b):        
        fake_b = 0.5*(self.G_A2B(real_a) + 1.0)
        fake_a = 0.5*(self.G_B2A(real_b) + 1.0)
        
        return fake_a,fake_b


# In[3]:


class ClassificationStatsManager(nt.StatsManager):
    def __init__(self):
        super(ClassificationStatsManager, self).__init__()
        
    def init(self):
        super(ClassificationStatsManager, self).init()
        self.running_loss_G = 0
        self.running_loss_D_A = 0
        self.running_loss_D_B = 0
    
    def accumulate(self, loss_G, loss_D_A, loss_D_B):
        super(ClassificationStatsManager, self).accumulate(loss_G, loss_D_A, loss_D_B)
        self.running_loss_G += loss_G
        self.running_loss_D_A += loss_D_A
        self.running_loss_D_B += loss_D_B

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        loss_G = self.running_loss_G / self.number_update 
        loss_D_A = self.running_loss_D_A / self.number_update 
        loss_D_B = self.running_loss_D_B / self.number_update
        return { 'G loss' : loss_G, 'D_A loss' : loss_D_A, 'D_B loss' : loss_D_B}


# In[43]:


lr = 1e-3
net = C_GAN()
net = net.to(device)

optimizer_G = torch.optim.Adam(itertools.chain(net.G_A2B.parameters(), 
                                               net.G_B2A.parameters()), lr=lr)
optimizer_D_A = torch.optim.Adam(net.D_A.parameters(), lr=lr)
optimizer_D_B = torch.optim.Adam(net.D_B.parameters(), lr=lr)

stats_manager = ClassificationStatsManager()
exp1 = nt.Experiment(net, train_set, val_set, 
                     optimizer_G, optimizer_D_A,optimizer_D_B,
                     stats_manager, output_dir="CGAN",batch_size=1)


# In[8]:


exp1.run(num_epochs=50)


# In[39]:


def GetResults(net,test_set):
    test_loader = td.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)
    net.eval()
    
    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')
        
    with torch.no_grad():
        i=0
        for real_a,real_b in test_loader:
            real_a, real_b = real_a.to(net.device), real_b.to(net.device)
            fake_a,fake_b = net(real_a, real_b)
            # Save image files
            tv.utils.save_image(fake_a, 'output/A/%04d.png' % (i+1))
            tv.utils.save_image(fake_b, 'output/B/%04d.png' % (i+1))
            i+=1
            
            print('Generated images %04d of %04d' % (i, len(test_loader)))


# In[40]:


GetResults(net,test_set)

