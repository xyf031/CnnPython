#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

# from datasets.factory import get_imdb
import mrc_data

import caffe
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default="models/mrcCaffeNet/solverA.prototxt", type=str)
    # parser.add_argument('--imdb', dest='imdb_name',
    #                     help='dataset to train on',
    #                     default='voc_2007_trainval', type=str)
    parser.add_argument('--folder', dest='folder_path',
                        help='dataset to train on',
                        default='/home/xyf/ssd/histeqMRC/gammas-lowpass', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=3000, type=int)  # 1 iter = 2 seconds, 3000 iter = 100min, 10000=5.5h
    
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default="data/imagenet_models/CaffeNet.v2.caffemodel", type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    if args.gpu_id is None or args.gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    imdb = mrc_data.mrc_data(args.folder_path)  # <--- Here!
    print 'PYTHON----------------------------------------Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    cfg.EXP_DIR = '20161207'
    output_dir = get_output_dir(imdb, None)
    print 'PYTHON----------------------------------------Output will be saved to `{:s}`'.format(output_dir)

    cfg.TRAIN.SCALES = (3500,)
    cfg.TRAIN.MAX_SIZE = 4000
    
    # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    cfg.TRAIN.FG_THRESH = 0.5
    # Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
    cfg.TRAIN.BG_THRESH_HI = 0.5
    cfg.TRAIN.BG_THRESH_LO = 0.1
    # Overlap required between a ROI and ground-truth box in order for that ROI to
    # be used as a bounding-box regression training example
    cfg.TRAIN.BBOX_THRESH = 0.5

    cfg.TRAIN.SNAPSHOT_ITERS = 1000

    print('PYTHON----------------------------------------Using config:')
    pprint.pprint(cfg)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)


