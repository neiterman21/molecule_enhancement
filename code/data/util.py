import os
import math
import pickle
import random
import numpy as np

###################### get image path list ######################
IMG_EXTENSIONS = [ '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','ARW','mrc','mrcs']


def is_image_file(filename , data_type = None):
    if data_type is None:
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return any(filename.endswith(extension) for extension in [data_type])

def _get_paths_from_images(path,data_type):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname,data_type):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths( dataroot,data_type):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:     
        if data_type == 'mrc' or data_type == 'mrcs':
            paths = sorted(_get_paths_from_images(dataroot,data_type))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths