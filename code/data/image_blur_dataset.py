import torch
import torch.utils.data as data
import mrcfile
import data.util as util
import numpy as np

class MoleculeBlurDataset(data.Dataset):
    def __init__(self, opt):
        super(MoleculeBlurDataset, self).__init__()
        self.opt = opt
        self.paths = util.get_image_paths( opt['dataroot'],opt['data_type'])
        if (opt['debug']):
            self.paths = self.paths[:5]
        #print(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with mrcfile.open(self.paths[index]) as mrc:
            d = mrc.data/16383
        avg = np.average(d, axis=(0))
        name = self.paths[index].split('/')[-1].split('.')[0]
        return {'video' : d, 'avg' : avg, 'name' : name}