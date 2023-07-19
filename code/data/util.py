import os
import math
import pickle
import random
import numpy as np
import copy

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

class point():

    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __eq__(self, __o: object) -> bool:
        if self.x == __o.x and self.y == __o.y:
            return True
        return False

    def __str__(self):
        return "[{x},{y}]".format(x=self.x,y=self.y)

    def is_near(self,other,radios):
        if math.dist([self.x,self.y] , [other.x,other.y]) < radios:
            return True
        return False

class moleculeCoords():
    def __init__(self,spamreader = None, kp = None,min_dist=45 ):
        self.points = []
        self.min_dist = min_dist
        if spamreader is not None:
            spamreader = spamreader.tolist()
            for row in spamreader[0]:
                self.points.append(point(int(row[0]) , int(row[1])))
        else:
            if kp is not None:
                for c in kp:
                    self.points.append(point(int(c.pt[0]) , int(c.pt[1])))

    def __contains__(self, point):
        for p in self.points:
            if p.is_near(point, self.min_dist):
                return True
        return False
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __getitem__(self,idx):
        return self.points[idx]
    def __len__(self):
        return len(self.points)

    def to_numpy(self):
        points = []
        for p in self.points:
            points.append((int(p.x) , int(p.y)))
        return np.asarray(points)

    def save(self,file : str):
        xs = []
        ys = []
        for p in self.points:
            xs.append(p.x)
            ys.append(p.y)
        xs = np.array(xs)
        ys = np.array(ys)
        with open(file, 'wb') as f:
            np.save(f, xs)
            np.save(f, ys)

    def delet_point(self,point):
        self.points.remove(point)
    
    def delete_point_list(self,p_list):
        for p in p_list:
          self.delet_point(p)  

    def erase(self,idx):
        del self.points[idx]
    
    def res_matrix(self,other_):
        other = other_.copy()
        fp = 0
        hit = 0
        miss = 0
        false_positive = []
        for p in self.points:
            fake = True
            for i , _ in enumerate(other):
                if p.is_near(other[i], self.min_dist*3):
                    fake = False
                    hit +=1
                    other.erase(i)
                    break
            if fake:
                false_positive.append(p)
                fp +=1
        mol_false = moleculeCoords()
        mol_false.points = false_positive
        return {'hit' : hit , 'miss' : len(other) , 'fp' : fp, 'false_positive' : mol_false}

    def merge(self,other_):
        other = other_.copy()
        for p in other.points:
            insert=True
            for i , _ in enumerate(self.points):
                if p.is_near(self.points[i], int(self.min_dist*3)):
                    insert=False
                    break
            if insert:
                self.points.append(p)