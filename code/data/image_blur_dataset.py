import torch
import torch.utils.data as data
import mrcfile
import data.util as util
import numpy as np
import csv
import starfile
import os
from skimage.restoration import denoise_nl_means
import re

class MoleculeBlurDataset(data.Dataset):
    def __init__(self, opt):
        super(MoleculeBlurDataset, self).__init__()
        self.opt = opt
        self.paths = util.get_image_paths( opt['dataroot'],opt['data_type'])
        if (opt['debug']):
            self.paths = self.paths[:opt['debug_imgs']]
        #self.paths = self.paths[600:]
        #print(self.paths)

    def __len__(self):
        return len(self.paths)

    def getcords_coord(self, index):
        cord_file_path = self.paths[index][:-3] + "coord"
        points = []
        with open(cord_file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                points.append((int(row[0]) , int(row[1])))
        return np.asarray(points)
    
    def extract_digits(self,s):
        import re
        match = re.match(r'(\d{3,4})', s)
        return match.group(1) if match else ''

    def getcords_star(self, index):
        name = self.paths[index].split('/')[-1].split('.')[0]
        cord_file_path = self.extract_digits(name) + "_autopick.star"
        df = starfile.read(os.path.join(os.path.dirname(self.paths[index]), cord_file_path))
        return df.to_numpy()[:,:2]
    
    def get_ctf_params(self,index):
        name = self.paths[index].split('/')[-1].split('.')[0]
        ctf_file_name = self.extract_digits(name) + "_ctffind3.log"
        ctf_file_path = os.path.join(os.path.dirname(self.paths[index]), ctf_file_name)
        with open(ctf_file_path, 'r') as file:
            for line in file:
                # Check if the line contains 'Final Values'
                if 'Final Values' in line:
                    # Use regular expression to extract numbers
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                    # Extract the first three numbers and convert to float
                    if len(numbers) >= 3:
                        DefocusU, DefocusV , DefocusAngle = map(float, numbers[:3])
                        break  # Stop reading further as we found the relevant line
        
        return np.array([DefocusU/10, DefocusV , np.radians(DefocusAngle)],dtype='f')


    def __getitem__(self, index):
        patch_kw = dict(patch_size=5,patch_distance=6)
        with mrcfile.mmap(self.paths[index],mode='r') as mrc:
            d = mrc.data/16383
            
           # d = denoise_nl_means(d,h =2, fast_mode=True, **patch_kw)
            d = d.astype('float32')

        
        if self.opt['data_type'] == 'mrc':
            d = np.expand_dims(d, axis=0)
        avg = np.average(d, axis=(0))
        name = self.paths[index].split('/')[-1].split('.')[0]
        ###
        if self.opt['cord_type'] == 'star':
            gt = self.getcords_star(index)
        elif self.opt['cord_type'] == None:
            gt = np.zeros(1)
        else:
            gt = self.getcords_coord(index)
        
        ctf_params = self.get_ctf_params(index)
        return {'video' : d, 'avg' : avg, 'name' : name , 'GT' : gt, 'ctf':  ctf_params}