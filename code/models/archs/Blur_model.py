import functools
#from tkinter import X
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy
import numpy as np

        

class BlurModel(nn.Module):
    
    def __init__(self, opt):
        super(BlurModel, self).__init__()
        self.opt = opt
        self.blur_k = torch.full((1,1,self.opt['conv_size'],self.opt['conv_size']), 1/(self.opt['conv_size']**2)).cuda()

        self.th1 = self.opt['th1']
        self.th2 = self.opt['th2']
        self.maxPull = nn.MaxPool2d(self.opt['morphology_size'],stride=1,padding=int(self.opt['morphology_size']/2))
        self.minPull = self.min_pool
        self.gaussiankernel = self.Gaussiankernel(self.opt['gauss_blur_size'])
    
    def forward(self, x):
        N ,c , h, w = x.shape
    
        output = F.conv2d(x.view(-1,1,h,w), self.blur_k , padding='same').view(N,c,h,w)
        #output = F.conv2d(output.view(-1,1,h-9,w-9), self.blur_k).view(N,c,h-18,w-18)
        threashed = torch.where(output > self.th1, 1. , 0.)
        #print("starting mean:" ,  torch.mean(torch.mean(threashed)))
        while(torch.mean(threashed) < 0.84 ):
            self.th1 = self.th1-0.0005
            threashed = torch.where(output > self.th1, 1. , 0.)
            #print("increasing mean:" ,  torch.mean(torch.mean(threashed)), 'self.th1:',self.th1)
        while(torch.mean(threashed) > 0.86 ):
            self.th1 = self.th1+0.0005
            threashed = torch.where(output > self.th1, 1. , 0.)
            #print("decreasing mean:" ,  torch.mean(torch.mean(threashed)), 'self.th1:',self.th1)
        #print('final:' , torch.mean(torch.mean(threashed)))  
        #output = torch.where(output <= self.th1, 0. , output.type(torch.DoubleTensor))
        output = self.minPull(self.maxPull(threashed)).type(torch.cuda.DoubleTensor)
        return output
        #output = F.conv2d(output.view(-1,1,h,w), self.gaussiankernel , padding='same').view(N,c,h,w)
        #output = torch.where(output > self.th2, 1. , 0.)
        #print(output)
        #return output

    def Gaussiankernel(self,ksize=51):
        n= np.zeros((ksize,ksize))
        k = scipy.ndimage.gaussian_filter(n,sigma=0)
        out = torch.from_numpy(k).cuda()
        return torch.unsqueeze(torch.unsqueeze(out,0),0)

    def min_pool(self,x):
        return -self.maxPull(-x)