#!/usr/bin/env python
# coding: utf-8

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
#from utils import weights_init_normal
import itertools
import time
import models as models
from utils import *


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


dataset_root_dir="/home/ayon.biswas/public_html/CycleGan/vangogh2photo/"


# In[4]:


# Dataset loader
train_set = ImageDataset(dataset_root_dir, image_size=256, unaligned=True, mode='train')
test_set = ImageDataset(dataset_root_dir, image_size=256, unaligned=False, mode='test')


# In[6]:




# In[7]:


class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.MSE_Loss = nn.MSELoss()
        self.L1_Loss = nn.L1Loss()
       
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
#         fake_b = 0.5*(self.G_A2B(real_a) + 1.0)
#         fake_a = 0.5*(self.G_B2A(real_b) + 1.0)
        fake_b = self.G_A2B(real_a)
        fake_a = self.G_B2A(real_b)
        return fake_a,fake_b


# In[8]:


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


# In[9]:


def norm_im(image, ax=plt):
#     image = image.to('cpu').numpy()
#     image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
#     image = (image + 1) / 2
#     image[image < 0] = 0
#     image[image > 1] = 1
    
# #     h = ax.imshow(image)
# #     ax.axis('off')
#     return image

    transform = tv.transforms.Compose([ tv.transforms.Normalize((-0.5,-0.5,-0.5), (2,2,2)) ])
    # In[10]:
    return transform(image)


def plot(exp, fig, axes, A, B, visu_rate=2):
    if exp.epoch % visu_rate != 0:
        return
    with torch.no_grad():
        fake_a,fake_b = exp.net(A[np.newaxis].to(exp.net.device),B[np.newaxis].to(exp.net.device))
    
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    myimshow(A, ax=axes[0][0])
    axes[0][0].set_title('real A')
    
    myimshow(fake_b[0], ax=axes[0][1])
    axes[0][1].set_title('fake B')
    
    myimshow(B, ax=axes[1][0])
    axes[1][0].set_title('real B')
    
    myimshow(fake_a[0], ax=axes[1][1])
    axes[1][1].set_title('fake A')
    
    plt.tight_layout()
    fig.canvas.draw()

def GetResults(net,test_set,cnt,dir_):
    test_loader = td.DataLoader(test_set, batch_size=4, shuffle=False, drop_last=True, pin_memory=True)
    net.eval()
    
    # Create output dirs if they don't exist
    if not os.path.exists('output_final/{}/A2B'.format(dir_)):
        os.makedirs('output_final/{}/A2B'.format(dir_))
    if not os.path.exists('output_final/{}/B'.format(dir_)):
        os.makedirs('output_final/{}/B'.format(dir_))
    if not os.path.exists('output_final/{}/B2A'.format(dir_)):
        os.makedirs('output_final/{}/B2A'.format(dir_))
    if not os.path.exists('output_final/{}/A'.format(dir_)):
        os.makedirs('output_final/{}/A'.format(dir_))
        
        
    with torch.no_grad():
        i=0
        for real_a,real_b in test_loader:
            real_a, real_b = real_a.to(net.device), real_b.to(net.device)
            fake_a,fake_b = net(real_a, real_b)
            a2b2a,b2a2a  = net(fake_a,fake_b)
            # Save image files
            tv.utils.save_image(0.5*(a2b2a+1), 'output_final/%s/A2B/%04d.png' % (dir_,i+1))
            tv.utils.save_image(0.5*(real_b+1), 'output_final/%s/B/%04d.png' % (dir_, i+1))
            tv.utils.save_image(0.5*(b2a2a+1), 'output_final/%s/B2A/%04d.png' % (dir_, i+1))
            tv.utils.save_image(0.5*(real_a+1), 'output_final/%s/A/%04d.png' % (dir_ , i+1))
            i+=1
            
            print('Generated images %04d of %04d' % (i, len(test_loader)))
            if(i == cnt):
                break

lr = 0.001

net = C_GAN()
net = net.to(device)

optimizer_G = torch.optim.Adam(itertools.chain(net.G_A2B.parameters(), 
                                               net.G_B2A.parameters()), lr=lr)
optimizer_D_A = torch.optim.Adam(net.D_A.parameters(), lr=lr)
optimizer_D_B = torch.optim.Adam(net.D_B.parameters(), lr=lr)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

lr_scheduler = [lr_scheduler_G, lr_scheduler_D_A,lr_scheduler_D_B]
stats_manager = ClassificationStatsManager()
exp1 = nt.Experiment(net, train_set, test_set, 
                     optimizer_G, optimizer_D_A,optimizer_D_B,lr_scheduler,stats_manager, output_dir="CGAN_wout_Idloss",batch_size=5)

# print(exp1.history[-1:-3:-1])





# In[ ]:
start_t = time.time()

# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,7))
# val_a,val_b=test_set[10]
# exp1.run(200,lr_scheduler)

# np.save("test_orig_dict.npy",exp1.history)
# In[ ]:

print("duration",time.time()-start_t)


            
# In[ ]:


GetResults(net,test_set,50,"van_gogh_wid_a2b2a")

# lr = 1.000000000000001e-05
# net = C_GAN()
# net = net.to(device)

# optimizer_G = torch.optim.Adam(itertools.chain(net.G_A2B.parameters(), 
#                                                net.G_B2A.parameters()), lr=lr)
# optimizer_D_A = torch.optim.Adam(net.D_A.parameters(), lr=lr)
# optimizer_D_B = torch.optim.Adam(net.D_B.parameters(), lr=lr)

# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

# lr_scheduler = [lr_scheduler_G, lr_scheduler_D_A,lr_scheduler_D_B]
# stats_manager = ClassificationStatsManager()
# exp2 = nt.Experiment(net, train_set, test_set, 
#                      optimizer_G, optimizer_D_A,optimizer_D_B,lr_scheduler,stats_manager, output_dir="CGAN_wout_Idloss",batch_size=8)

# # print(exp1.history[-1:-3:-1])





# # In[ ]:
# start_t = time.time()

# # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,7))
# # val_a,val_b=test_set[10]
# exp2.run(200,lr_scheduler)

# GetResults(net,test_set,50,"van_gogh_v2")
# # In[ ]:



# lr =  1.000000000000001e-05
# net = C_GAN()
# net = net.to(device)

# optimizer_G = torch.optim.Adam(itertools.chain(net.G_A2B.parameters(), 
#                                                net.G_B2A.parameters()), lr=lr)
# optimizer_D_A = torch.optim.Adam(net.D_A.parameters(), lr=lr)
# optimizer_D_B = torch.optim.Adam(net.D_B.parameters(), lr=lr)

# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

# lr_scheduler = [lr_scheduler_G, lr_scheduler_D_A,lr_scheduler_D_B]
# stats_manager = ClassificationStatsManager()
# exp3 = nt.Experiment(net, train_set, test_set, 
#                      optimizer_G, optimizer_D_A,optimizer_D_B,lr_scheduler,stats_manager, output_dir="CGAN_cez",batch_size=5)

# # print(exp1.history[-1:-3:-1])





# # In[ ]:
# start_t = time.time()

# # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,7))
# # val_a,val_b=test_set[10]
# exp3.run(200,lr_scheduler)

# GetResults(net,test_set,50,"cezanne")


# # In[11]:
# lr = 0.001
# net = C_GAN()
# net = net.to(device)

# optimizer_G = torch.optim.Adam(itertools.chain(net.G_A2B.parameters(), 
#                                                net.G_B2A.parameters()), lr=lr)
# optimizer_D_A = torch.optim.Adam(net.D_A.parameters(), lr=lr)
# optimizer_D_B = torch.optim.Adam(net.D_B.parameters(), lr=lr)

# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

# lr_scheduler = [lr_scheduler_G, lr_scheduler_D_A,lr_scheduler_D_B]
# stats_manager = ClassificationStatsManager()
# exp4 = nt.Experiment(net, train_set, test_set, 
#                      optimizer_G, optimizer_D_A,optimizer_D_B,lr_scheduler,stats_manager, output_dir="CGAN_ukiyo",batch_size=1)

# # print(exp1.history[-1:-3:-1])


# # In[ ]:
# start_t = time.time()

# # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,7))
# # val_a,val_b=test_set[10]
# exp4.run(100,lr_scheduler)

# GetResults(net,test_set,50,"ukiyo")



