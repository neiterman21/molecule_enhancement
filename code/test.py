import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import utils.util as util

import options.options as option
from data import create_dataset, create_dataloader
from data.util import moleculeCoords
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

hit = 0
mis = 0
fp = 0
need_gt = True
for data in test_loader:
    #if data['GT'] == np.zeros(1):
    need_gt = True
    model.feed_data(data)

    logger.info('\Running [{:s}]...'.format(data['name'][0]))
    model.test()
    visuals = model.get_current_visuals()
    
    sr_img = util.tensor2img(visuals['rlt'])  # uint8
    # save images
    #print("printing!!!" , dataset_dir,data['name'][])
    image_dir = osp.join(dataset_dir, data['name'][0])
    if opt['save_res'] or opt['save_star']:
        util.mkdir(image_dir)
    avg_img = (data['avg'].cpu().data.numpy()*255*10).astype('uint8')[0]
    frames = (data['video'].cpu().data.numpy()*255*10).astype('uint8')[0]

    if opt['save_res']:
        util.save_img(avg_img, osp.join(image_dir, 'avg.jpg'))
    video_hit = 0
    video_mis = 0
    video_fp = 0  
    if need_gt:
        gt = moleculeCoords(spamreader = data['GT'])
    from_all = moleculeCoords()
    all_kps = []
    for i , im in enumerate(sr_img):
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(image_dir, str(i) + suffix + '.jpg')
            save_img_path_org = osp.join(image_dir, str(i) + suffix + '_org.jpg')
        else:
            save_img_path = osp.join(image_dir, str(i) + '.jpg')
            save_img_path_org = osp.join(image_dir, str(i) + '_org.jpg')
        
        # Save SR images for reference
        sr_img_ = (sr_img[i]*255).astype('uint8')
        org_frame = (frames[i]).astype('uint8')
        
        #keypoints = detector.detect(sr_img_)
        #all_kps += keypoints
        #coords = moleculeCoords(kp=keypoints)
        coords = visuals['coords'][i]
        from_all.merge(coords)
        if need_gt:
            score = coords.res_matrix(gt)
            logger.info(score)

            video_hit += score['hit']
            video_mis += score['miss']
            video_fp += score['fp']
            avg_w_bullets = util.draw_keypoints2(data['GT'],avg_img,color=(255,0,0))
        else:
            avg_w_bullets = avg_img.copy()
        
        if opt['save_binary'] :  
            if need_gt:       
                sr_img_ = util.draw_keypoints2(data['GT'],sr_img_)
                #org_frame = util.draw_keypoints2(data['GT'],org_frame)
            sr_img_ = util.draw_keypoints2(coords.to_numpy(),sr_img_,color=(0,0,255),size=8,rec=True)
            org_frame = util.draw_keypoints2(coords.to_numpy(),org_frame,color=(0,0,255),size=8,rec=True)
            if i != 0 :
                org_frame_i = util.draw_keypoints2(coords.to_numpy(),org_frame_i,color=(255,0,0),size=8,rec=True)
                

        else:
            sr_img_ = util.draw_keypoints2(coords.to_numpy(),avg_w_bullets,color=(0,0,255),size=12,rec=True)
        
        if opt['save_res']:
            util.save_img(sr_img_.astype('uint8'), save_img_path,use_PIL=False)
            if opt['save_binary']:
                util.save_img(org_frame.astype('uint8'), save_img_path_org,use_PIL=False)
        if i == 0: org_frame_i = org_frame
        if i != 15:
            if opt['save_res']:
                util.save_img(org_frame_i.astype('uint8'), save_img_path_org,use_PIL=False) 
            org_frame_i = org_frame
            #avg_post += sr_img_
    #logger.info('\ rates: hit [{:s}] miss [{:s}] fp [{:s}]...'.format(str(video_hit/(video_hit+video_mis)),str(video_mis/(video_hit+video_mis)),str(video_fp/(video_hit+video_mis))))
    #stats taken from all frames
    ####### Gil debug
    avg_w_bullets = util.draw_keypoints2(data['GT'],avg_img)
    avg_w_bullets = util.draw_keypoints2(all_kps,avg_w_bullets,color=(0,0,255),size=2)
    util.save_img(avg_w_bullets, osp.join(image_dir, 'all_bullets.jpg'),use_PIL=False)
    #######
    if need_gt:
        score_all = from_all.res_matrix(gt)
        logger.info('from all ' + opt['name'] + '\hit [{:d}] miss [{:d}] fp [{:d}]...'.format(score_all['hit'],score_all['miss'],score_all['fp']))
        logger.info('from all ' + opt['name'] + ' \ rates: hit [{:.3f}] miss [{:.3f}] fp [{:.3f}]...'.format(score_all['hit']/(score_all['hit']+score_all['miss']),score_all['miss']/(score_all['hit']+score_all['miss']),score_all['fp']/(score_all['hit']+score_all['miss'])))
        hit += score_all['hit']
        mis += score_all['miss']
        fp += score_all['fp']
    
        
    sr_img_ = util.draw_keypoints2(from_all.to_numpy(),avg_img,color=(0,0,255),size=8,rec=True)
    if need_gt:
        sr_img_ = util.draw_keypoints2(data['GT'],sr_img_)
    if opt['save_res']:
        util.save_img(sr_img_, osp.join(image_dir, 'from_all.jpg'),use_PIL=False)
    if opt['save_star']:
        save_star_path = osp.join(image_dir, data['name'][0] + '.star')
        util.save_star(from_all,save_star_path)
    #######
if need_gt:
    logger.info('Total hit [{:d}] miss [{:d}] fp [{:d}]...'.format(hit,mis,fp))
    logger.info('Total rates ' + opt['name'] + ': hit [{:.3f}] miss [{:.3f}] fp [{:.3f}]...'.format(hit/(hit+mis),mis/(hit+mis),fp/(hit+mis)))
    #avg_post /= len(sr_img)
    #util.save_img(avg_img, osp.join(image_dir, 'avg_post.jpg'))
