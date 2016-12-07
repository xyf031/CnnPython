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

# from datasets.factory import get_imdb
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
    # parser.add_argument('--imdb', dest='imdb_name',
    #                     help='dataset to test',
    #                     default='voc_2007_test', type=str)
    parser.add_argument('--folder', dest='folder_path',
                        help='dataset to train on',
                        default='/home/xyf/ssd/histeqMRC/gammas-lowpass', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default="models/mrcCaffeNet/testA.prototxt", type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default="output/store/gammas-histeq/mrc_caffenet_iter_3000.caffemodel", type=str)
    
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
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

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    # set up caffe
    if args.gpu_id is None or args.gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = mrc_data.mrc_data(args.folder_path)
    imdb.competition_mode(args.comp_mode)

    cfg.TEST.SCALES = (3500,)
    cfg.TEST.MAX_SIZE = 4000

    # Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
    cfg.TEST.NMS = 0.3
    # Test using bounding-box regressors
    cfg.TEST.BBOX_REG = True
    # Experimental: treat the (K+1) units in the cls_score layer as linear
    # predictors (trained, eg, with one-vs-rest SVMs).
    # cfg.TEST.SVM = False
    cfg.EXP_DIR = '20161206'

    print('PYTHON----------------------------------------Using config:')
    pprint.pprint(cfg)

    test_net(net, imdb)
