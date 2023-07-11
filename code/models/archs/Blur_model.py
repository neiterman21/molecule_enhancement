import functools
#from tkinter import X
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy
import numpy as np
import sys
import models.archs.blob_detection as blob_detection
from  data.util import moleculeCoords
from  scipy.ndimage import minimum_filter
        

class BlurModel(nn.Module):
    
    def __init__(self, opt):
        super(BlurModel, self).__init__()
        self.opt = opt
        self.blur_k = torch.full((1,1,self.opt['conv_size'],self.opt['conv_size']), 1/(self.opt['conv_size']**2)).cuda()
        #self.detector = blob_detection.create_blob_detector(self.opt['minArea'])
        self.th1 = self.opt['th1']
        self.mean = self.opt['patch_mean']
        self.maxPull = nn.MaxPool2d(self.opt['morphology_size'],stride=1,padding=int(self.opt['morphology_size']/2))
        self.minPull = self.min_pool
        self.sqrt_frags = self.opt['sqrt_frags']
        self.voting_th = self.opt['voting_th']
        self.frame_patches = [0,1,2,3,4,5,6,7,8,15,16,23,24,31,32,39,40,47,48,55,56,57,58,59,60,61,62,63]
        self.double_votes = self.opt['double_vots']

    def get_blobs(self,x):
        x = (x.data.numpy()*255).astype('uint8')
        coords = []
        coords_larg = []
        for frame in x[0]:
            coords.append(moleculeCoords(kp=blob_detection.get_blobs(frame,self.opt['minArea'])))
            if self.double_votes: coords_larg.append(moleculeCoords(kp=blob_detection.get_blobs(frame,2*self.opt['minArea'])))
        for cord_set , cord_set_larg in zip(coords,coords_larg):
            to_delete = []
            for point in cord_set.points:
                vote = 0
                for cord in coords:
                    
                    if point in cord:
                        vote+=1
                if self.double_votes:
                    for cord in coords_larg:                   
                        if point in cord:
                            vote+=2
                if vote < self.voting_th:
                    to_delete.append(point)
            cord_set.delete_point_list(to_delete)
        return coords
    
    def forward(self, x):
        N ,c , h, w = x.shape
        #x = F.conv2d(x.view(-1,1,h,w), self.blur_k , padding='same').view(N,c,h,w)
        output = F.unfold(x, kernel_size=int(h/self.sqrt_frags), stride=int(h/self.sqrt_frags)) 
        output = output.view(N,  int(h/self.sqrt_frags), int(w/self.sqrt_frags),c*(self.sqrt_frags**2),)
        output = output.permute(0, 3, 1, 2)
        for i in range(self.sqrt_frags**2):
            if i in self.frame_patches:
                f_bias = 0.05
            else:
                f_bias = 0
            threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
            #print("starting mean:" ,  torch.mean(torch.mean(threashed)))
            while(torch.mean(threashed) < self.mean+0.02-f_bias):
                self.th1 = self.th1-0.00005
                threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
                #print("increasing mean:" ,  torch.mean(torch.mean(threashed)), 'self.th1:',self.th1)
            while(torch.mean(threashed) > self.mean - 0.02 -f_bias):
                self.th1 = self.th1+0.00005
                threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
            output[0,i] = threashed
        output = output.permute(0, 2, 3, 1)
        output = output.view(1,-1,self.sqrt_frags**2)
        output = F.fold(output, kernel_size=int(h/self.sqrt_frags),stride=int(h/self.sqrt_frags) ,output_size=(h,w))

        output = self.minPull(self.maxPull(output)).type(torch.cuda.DoubleTensor)
        cordes = self.get_blobs(output.clone().cpu())
        #self.find_min(conved.clone().cpu())
        
        return output , cordes

    def min_pool(self,x):
        return -self.maxPull(-x)