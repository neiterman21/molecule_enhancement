import torch
import torch.utils.data as data
import mrcfile
import data.util as util
import numpy as np
import csv
import starfile
import os
from skimage.restoration import denoise_nl_means

class MoleculeBlurDataset(data.Dataset):
    def __init__(self, opt):
        super(MoleculeBlurDataset, self).__init__()
        self.opt = opt
        self.paths = util.get_image_paths( opt['dataroot'],opt['data_type'])
        if (opt['debug']):
            self.paths = self.paths[:opt['debug_imgs']]
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

    def getcords_star(self, index):
        name = self.paths[index].split('/')[-1].split('.')[0]
        cord_file_path = name[:3] + "_autopick.star"
        df = starfile.read(os.path.join(os.path.dirname(self.paths[index]), cord_file_path))
        return df.to_numpy()[:,:2]

    def __getitem__(self, index):
        patch_kw = dict(patch_size=5,patch_distance=6)
        with mrcfile.mmap(self.paths[index],mode='r') as mrc:
            d = mrc.data/16383
            
            d = denoise_nl_means(d,h =2, fast_mode=True, **patch_kw)
            d = d.astype('float32')

        
        #if self.opt['data_type'] == 'mrc':
        #    d = np.expand_dims(d, axis=0)
        avg = np.average(d, axis=(0))
        name = self.paths[index].split('/')[-1].split('.')[0]
        ###
        if self.opt['cord_type'] == 'star':
            gt = self.getcords_star(index)
        elif self.opt['cord_type'] == None:
            gt = np.zeros(1)
        else:
            gt = self.getcords_coord(index)
        
        return {'video' : d, 'avg' : avg, 'name' : name , 'GT' : gt }