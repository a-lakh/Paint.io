import glob
import random
import os

import numpy as np
import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import my_nntools as nt
import models as models

class ImageDataset(td.Dataset):
    def __init__(self, root, image_size=512, unaligned=False, mode='train'):

        self.transform = tv.transforms.Compose([ tv.transforms.Resize(int(image_size*1.12), Image.BICUBIC), 
                tv.transforms.RandomCrop(image_size), 
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
            
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
            

        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def init_parameters(m):
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)
    '''        
    if (isinstance(m, nn.Conv2d)):# or isinstance(m, nn.InstanceNorm2d)):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias.data)     #Can Change This

class ReplayBuffer():
    def __init__(self, max_size=50):     #Can Change This
        assert (max_size > 0), 'Buffer Empty'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
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
    


