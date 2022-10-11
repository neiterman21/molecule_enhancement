import functools
from tkinter import X
import torch.nn as nn
import torch.nn.functional as F
import torch


class BlurModel(nn.Module):
    
    def __init__(self):
        super(BlurModel, self).__init__()
        self.blur_k = torch.full((1,1,100,100), 0.0001)
    
    def forward(self, x):
        N ,c , h, w = x.shape
   
        output = F.conv2d(x.view(-1,1,h,w), self.blur_k).view(N,c,h-99,w-99)
        #output = F.conv2d(output.view(-1,1,h-9,w-9), self.blur_k).view(N,c,h-18,w-18)
        output = torch.where(output > 0.129, 1. , output.type(torch.DoubleTensor))
        return output
