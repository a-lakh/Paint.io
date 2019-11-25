import glob
import random
import os

import numpy as np
import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv
from PIL import Image

class ImageDataset(td.Dataset):
    def __init__(self, root, image_size=512, unaligned=False, mode='train'):

        self.transform = tv.transforms.Compose([ tv.transforms.Resize(int(image_size*1.12), Image.BICUBIC), 
                tv.transforms.RandomCrop(image_size), 
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

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