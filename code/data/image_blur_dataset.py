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

        #pattern = r'^\d{4}'
        no_gt = [600,597,586,598,594,596,589,592,588,595,590,599,593,587,591]
        path_less = []
        #for file in self.paths:
            #if re.match(pattern, os.path.basename(file)):
            #    path_less.append(file)
        #    if int(os.path.basename(file)[:3]) in no_gt:
        #        path_less.append(file)
            
        self.paths = [elem for elem in self.paths if elem not in path_less]

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
        cord_file_path = name + ".star"
        #cord_file_path = self.extract_digits(name) + "_autopick.star"
        #print(os.path.join(os.path.dirname(self.paths[index]), cord_file_path))
        df = starfile.read(os.path.join(os.path.dirname(self.paths[index]), cord_file_path))
        return df.to_numpy()[:,2:4].astype(np.float32)
    
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
        with mrcfile.mmap(self.paths[index],mode='r') as mrc:
            d = mrc.data/16#383    
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
        ctf_params = None
        #ctf_params = self.get_ctf_params(index)
        #print(gt,type(gt))
        output = {'video' : d , 'avg' : avg ,'GT' : gt ,'name' : name }# , 'ctf':  ctf_params}
  
        return output