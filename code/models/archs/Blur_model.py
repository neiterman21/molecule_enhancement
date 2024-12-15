import functools
#from tkinter import X
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy
import numpy as np
import sys
import models.archs.blob_detection as blob_detection
import models.archs.ctf_corection_pytorch as ctf_corect
from  data.util import moleculeCoords
from  scipy.ndimage import minimum_filter
from models.archs.arch_utils import *

class BlurModel(nn.Module):
    
    def __init__(self, opt):
        super(BlurModel, self).__init__()
        self.opt = opt
        self.blur_k = torch.full((1,1,self.opt['conv_size'],self.opt['conv_size']), 1/(self.opt['conv_size']**2)).cuda()
        #self.detector = blob_detection.create_blob_detector(self.opt['minArea'])
        self.th1 = self.opt['th1']
        self.mean = self.opt['patch_mean']
        self.maxPull = nn.MaxPool2d(self.opt['morphology_size'],stride=1,padding=int((self.opt['morphology_size'])/2))
        self.minPull = self.min_pool
        self.sqrt_frags = self.opt['sqrt_frags']
        self.voting_th = self.opt['voting_th']
        self.frame_patches = [0,1,2,3,4,5,6,7,8,15,16,23,24,31,32,39,40,47,48,55,56,57,58,59,60,61,62,63]
        self.double_votes = self.opt['double_vots']

        self.Cs = torch.tensor(2.0).cuda()  # In mm
        self.pixA = torch.tensor(1.34).cuda()  # In Angstrom
        self.AmplitudeContrast = torch.tensor(0.07).cuda()
        self.voltage = torch.tensor(300).cuda()
        self.n = torch.tensor((4096,4096)).cuda()

    def get_blobs(self,x):
        x = (x.data.numpy()*255).astype('uint8')
        coords = []
        coords_larg = []
        for frame in x[0]:
            coords.append(moleculeCoords(kp=blob_detection.get_blobs(frame,self.opt['minArea'])))
            if self.double_votes: coords_larg.append(moleculeCoords(kp=blob_detection.get_blobs(frame,2*self.opt['minArea'])))
        i=0
        for cord_set in coords: #, cord_set_larg in zip(coords,coords_larg):
            to_delete = []
            for point in cord_set.points:
                vote = 0
                for cord in coords:                   
                    if point in cord:
                        vote+=1
                if self.double_votes:
                    for cord in coords_larg:                   
                        if point in cord:
                            vote+=1
                
                if vote < self.voting_th:
                    to_delete.append(point)
            coords[i].delete_point_list(to_delete)
            i+=1
        return coords
    
    def corect_ctf(self,x,ctf_params):
        N ,c , h, w = x.shape
        for i in range(c):
            x[0,i] = ctf_corect.phase_flip(x[0,i], torch.tensor((h,w),device=x.device), self.voltage, ctf_params[0,0], ctf_params[0,1], ctf_params[0,2], self.Cs, self.pixA, self.AmplitudeContrast)**2
        return x
    
    def image_to_patches(self,image, patch_height, patch_width):
        """
        Convert a 3D image tensor to a 4D tensor of single-channel patches.

        Args:
        image (torch.Tensor): The image tensor of shape (C, H, W).
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.

        Returns:
        torch.Tensor: A 4D tensor of shape (num_patches, C, patch_height, patch_width).
        """
        N,C, H, W = image.shape

        # Reshape into a 5D tensor of (C, num_patches_height, patch_height, num_patches_width, patch_width)
        patches = image[0].unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
        
        # Reshape into a 4D tensor where each patch is separated and channel is second dimension
        patches = patches.contiguous().view(-1, 1, patch_height, patch_width)

        return patches

    def patches_to_image(self,patches, original_shape, patch_height, patch_width):
        """
        Reconstruct the original image from its patches.

        Args:
        patches (torch.Tensor): The 4D tensor of patches.
        original_shape (tuple): The shape of the original image (C, H, W).
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.

        Returns:
        torch.Tensor: The reconstructed image of shape (C, H, W).
        """
        C, H, W = original_shape
        # Calculate how many patches there are along each dimension
        num_patches_height = H // patch_height
        num_patches_width = W // patch_width

        # Reshape the patches tensor to a 5D tensor before folding
        patches_reshaped = patches.view(C, num_patches_height, num_patches_width, patch_height, patch_width)

        # Fold the patches back into the original image shape
        reconstructed = patches_reshaped.permute(0, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(1,C, H, W)

        return reconstructed

    def forward(self, x , ctf_params=None):
        if ctf_params is not None:
            x = self.corect_ctf(x,ctf_params)
        N ,c , h, w = x.shape
        x = F.conv2d(x.view(-1,1,h,w), self.blur_k , padding='same').view(N,c,h,w)
        #save_tensor_as_image(x[0,0], 'frame0_conved.png')
        #output = F.unfold(x, kernel_size=int(h/self.sqrt_frags), stride=int(h/self.sqrt_frags)) 
        #output = output.view(N,  int(h/self.sqrt_frags), int(w/self.sqrt_frags),c*(self.sqrt_frags**2),)
        output = self.image_to_patches(x,int(h/self.sqrt_frags),int(h/self.sqrt_frags))
        output = output.permute(1, 0, 2, 3)
        for i in range(c*self.sqrt_frags**2):
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
            while(torch.mean(threashed) > self.mean - 0.02 ):
                self.th1 = self.th1+0.000005
                threashed = torch.where(output[0,i] > self.th1, 1. , 0.)
            output[0,i] = threashed
        output = output.permute(1, 0, 2, 3)
        output = self.patches_to_image(output,(c , h, w),int(h/self.sqrt_frags),int(h/self.sqrt_frags))
        #output = output.view(1,-1,self.sqrt_frags**2)
        #output = F.fold(output, kernel_size=int(h/self.sqrt_frags),stride=int(h/self.sqrt_frags) ,output_size=(h,w))
        #save_tensor_as_image(output[0,0], 'frame0_threshed.png')
        #save_tensor_as_image(output[0,1], 'frame1_threshed.png')
        #save_tensor_as_image(output[0,2], 'frame2_threshed.png')
        output = self.minPull(self.maxPull(output)).type(torch.cuda.DoubleTensor)
        cordes = self.get_blobs(output.clone().cpu())
        #self.find_min(conved.clone().cpu())
        #save_tensor_as_image(output[0,0], 'frame0_morfed.png')
        #save_tensor_as_image(output[0,1], 'frame1_morfed.png')
        #save_tensor_as_image(output[0,2], 'frame2_morfed.png')
        return output , cordes

    def min_pool(self,x):
        return -self.maxPull(-x)