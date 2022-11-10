import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import utils.util as util

import options.options as option
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

#creat blob detection
detector = util.create_blob_detector()
for data in test_loader:
    model.feed_data(data)

    logger.info('\Running [{:s}]...'.format(data['name'][0]))
    model.test()
    visuals = model.get_current_visuals()

    sr_img = util.tensor2img(visuals['rlt'])  # uint8
    # save images
    #print("printing!!!" , dataset_dir,data['name'][])
    image_dir = osp.join(dataset_dir, data['name'][0])
    util.mkdir(image_dir)
    avg_img = (data['avg'].cpu().data.numpy()*255*2).astype('uint8')[0]
    avg_post = np.zeros_like(avg_img)
    
    util.save_img(avg_img, osp.join(image_dir, 'avg.jpg'))
    for i , im in enumerate(sr_img):
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(image_dir, str(i) + suffix + '.jpg')
        else:
            save_img_path = osp.join(image_dir, str(i) + '.jpg')
        
            # Save SR images for reference
            sr_img_ = (sr_img[i]*255).astype('uint8')
            keypoints = detector.detect(sr_img_)
            sr_img_ = util.draw_keypoints(keypoints,avg_img)
            util.save_img(sr_img_, save_img_path,use_PIL=False)
            #avg_post += sr_img_
    #avg_post /= len(sr_img)
    #util.save_img(avg_post, osp.join(image_dir, 'avg_post.jpg'))
