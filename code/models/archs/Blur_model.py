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
        self.detector = blob_detection.create_blob_detector(self.opt['minArea'])
        self.th1 = self.opt['th1']
        self.mean = self.opt['patch_mean']
        self.maxPull = nn.MaxPool2d(self.opt['morphology_size'],stride=1,padding=int(self.opt['morphology_size']/2))
        self.minPull = self.min_pool
        self.gaussiankernel = self.Gaussiankernel(self.opt['gauss_blur_size'])
        self.sqrt_frags = self.opt['sqrt_frags']
        self.voting_th = self.opt['voting_th']

    def get_blobs(self,x):
        x = (x.data.numpy()*255).astype('uint8')
        coords = []
        for frame in x[0]:
            coords.append(moleculeCoords(kp=self.detector.detect(frame)))
        for cord_set in coords:
            to_delete = []
            for point in cord_set.points:
                vote = 0
                for cord in coords:
                    if point in cord:
                        vote+=1
                if vote < self.voting_th:
                    to_delete.append(point)
            cord_set.delete_point_list(to_delete)
        return coords
    
    def forward(self, x):
        N ,c , h, w = x.shape
        #print('saving')
        #with open('video_0_full.npy', 'wb') as f:
        #    np.save(f, x.clone().cpu().numpy())
        conved = F.conv2d(x.view(-1,1,h,w), self.blur_k , padding='same').view(N,c,h,w)
        #with open('conved_0_full.npy', 'wb') as f:
        #    np.save(f, conved.clone().cpu().numpy())
        output = F.unfold(conved, kernel_size=int(h/self.sqrt_frags), stride=int(h/self.sqrt_frags)) 
        output = output.view(N,  int(h/self.sqrt_frags), int(w/self.sqrt_frags),c*(self.sqrt_frags**2),)
        output = output.permute(0, 3, 1, 2)
        for i in range(self.sqrt_frags**2):

            threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
            #print("starting mean:" ,  torch.mean(torch.mean(threashed)))
            while(torch.mean(threashed) < self.mean+0.02):
                self.th1 = self.th1-0.0005
                threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
                #print("increasing mean:" ,  torch.mean(torch.mean(threashed)), 'self.th1:',self.th1)
            while(torch.mean(threashed) > self.mean - 0.02 ):
                self.th1 = self.th1+0.0005
                threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
            output[0,i] = threashed
        output = output.permute(0, 2, 3, 1)
        output = output.view(1,-1,self.sqrt_frags**2)
        output = F.fold(output, kernel_size=int(h/self.sqrt_frags),stride=int(h/self.sqrt_frags) ,output_size=(h,w))
            #print("decreasing mean:" ,  torch.mean(torch.mean(threashed)), 'self.th1:',self.th1)
        #print('final:' , torch.mean(torch.mean(threashed)))  
        #output = torch.where(output <= self.th1, 0. , output.type(torch.DoubleTensor))
        output = self.minPull(self.maxPull(output)).type(torch.cuda.DoubleTensor)
        cordes = self.get_blobs(output.clone().cpu())
        #self.find_min(conved.clone().cpu())
        
        return output , cordes
        #output = F.conv2d(output.view(-1,1,h,w), self.gaussiankernel , padding='same').view(N,c,h,w)
        #output = torch.where(output > self.th2, 1. , 0.)
        #print(output)
        #return output
    def find_min(self,conved):
        print(conved.shape)
        for frame in conved[0]:
            with open('conved.npy','wb') as f:
                np.save(f,frame)
            minmap = minimum_filter(frame,size=40)
            print( minmap)
            exit(0)
    def Gaussiankernel(self,ksize=51):
        n= np.zeros((ksize,ksize))
        k = scipy.ndimage.gaussian_filter(n,sigma=0)
        out = torch.from_numpy(k).cuda()
        return torch.unsqueeze(torch.unsqueeze(out,0),0)

    def min_pool(self,x):
        return -self.maxPull(-x)