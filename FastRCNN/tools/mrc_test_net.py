#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list

import mrc_data

import caffe
import argparse
import pprint
import time, os, sys


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--folder', dest='folder_path', help='dataset to train on',
                        default='/home/xyf/ssd/histeqMRC/gammas-histeq', type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to test',
                        default="output/store/ga+tr-histeq/mrc_caffenet_iter_30000.caffemodel", type=str)

    parser.add_argument('--nms', dest='nms', help='cfg.TEST.NMS', default=0.85, type=float)
    parser.add_argument('--imgavg', dest='max_avg_img', help='cfg.TEST.MAX_AVG_IMAGE', default=1000, type=int)
    parser.add_argument('--imgmax', dest='max_per_img', help='cfg.TEST.MAX_PER_IMAGE', default=2000, type=int)
    
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default="models/mrcCaffeNet/testA.prototxt", type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print args

    if not os.path.exists(args.caffemodel):
        print('Waiting for {} to exist...'.format(args.caffemodel))
        exit(0)

    # set up caffe
    if args.gpu_id is None or args.gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        
    cfg.EXP_DIR = '20161212'
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]


    imdb = mrc_data.mrc_data(args.folder_path)

    cfg.TEST.SCALES = (3500,)
    cfg.TEST.MAX_SIZE = 4000

    # Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
    cfg.TEST.NMS = args.nms  # 0.3
    # Test using bounding-box regressors
    cfg.TEST.BBOX_REG = True
    # Experimental: treat the (K+1) units in the cls_score layer as linear
    # predictors (trained, eg, with one-vs-rest SVMs).
    # cfg.TEST.SVM = False
    
    cfg.TEST.MAX_AVG_IMAGE = args.max_avg_img  # 300
    cfg.TEST.MAX_PER_IMAGE = args.max_per_img  # 600

    print('PYTHON Test------------------------------------Using config:')
    pprint.pprint(cfg)

    test_net(net, imdb)
