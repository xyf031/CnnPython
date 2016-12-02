
import numpy as np
import os
import scipy.sparse
import scipy.io as sio
# import cPickle

import datasets.imdb

class mrc_data(datasets.imdb):
    def __init__(self, dataset_path):
        self._name = os.path.split(dataset_path)[1]  # Name of the protein.
        self._root_path = dataset_path  # Absolute path of the dataset folder.

        self._classes = ['__background__', 'particle']
        self._num_classes = len(self._classes)
        self._index_to_class = dict(zip(xrange(self._num_classes), self._classes))
        self._class_to_index = dict(zip(self._classes, xrange(self._num_classes)))

        self._image_index = []  # List of *.mrc.bmp file absolute path.
        self._star_index = []  # List of *.star file absolute path.

        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb

        # self._roidb_cache = "/home/xyf/hhh"
        self._box_side_length = 100
        self._read_ALL()  # <---------- Entrance


    def default_roidb(self, i):
        return 0

    def image_path_at(self, i):
        return self._image_index[i]

    def star_path_at(self, i):
        return self._star_index[i]

    def evaluate_detections(self, all_boxes, output_dir=None):
        return


    def _read_ALL(self):
        read_ALL(self._root_path)

    def read_ALL(self, dataset_path):
        mrc_txt_path = os.path.join(dataset_path, 'bmp.txt')
        star_txt_path = os.path.join(dataset_path, 'star.txt')
        mat_path = os.path.join(dataset_path, 'selective_search.mat')
        if not (os.path.exists(mrc_txt_path) and os.path.exists(star_txt_path) and os.path.exists(mat_path)):
            print "ERROR: read_ALL() fails. Some files do not exist."
            return

        # Read mrc.txt
        with open(mrc_txt_path) as f:
            self._image_index = [i.strip() for i in f.readlines()]
        # Read star.txt
        with open(star_txt_path) as f:
            self._star_index = [i.strip() for i in f.readlines()]

        gt_roidb = self.read_all_star()
        # Read selective_search.mat
        ss_roidb = self.read_selective_search_mat(gt_roidb, mat_path)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        self._roidb = roidb

    def read_selective_search_mat(self, all_star, mat_path):
        # Read selective_search.mat
        raw_data = sio.loadmat(mat_path)['boxes'].ravel()
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
            # Each item in raw_data is still a list, corresponds the Rois of the i-th pic.
            # Size of raw_data[i] is (roiNum, 4).
            # The columns of raw_data[i][j] is [rowBegin, colBegin, rowEnd, colEnd] in Matlab matrix.
            # In python indices, that means [yBegin, xBegin, yEnd, xEnd] + 1. (Matlab indices begin from 1).
        return self.create_roidb_from_box_list(box_list, all_star)

    def read_all_star(self):
        gt_roidb = [self.read_star_file(star_path) for star_path in self._star_index]
        return gt_roidb

    def read_star_file(self, file_path):
        # The box-border may exceed the pic! [0, max-pixel + 100/2]
        fStar = open(file_path)
        fLines = [ii.strip().split() for ii in fStar.readlines()]
        fStar.close()
    
        boxes = []
        for ii in fLines:
            try:
                x_center = float(ii[0])
                y_center = float(ii[1])
            except:
                print ii
                continue
            else:
                xmin = max(0, x_center - self._box_side_length / 2)
                xmax = x_center + self._box_side_length / 2
                ymin = max(0, y_center - self._box_side_length / 2)
                ymax = y_center + self._box_side_length / 2
                boxes.append([xmin, ymin, xmax, ymax])
    
        nBoxes = len(boxes)
        boxes = np.array(boxes, np.int32)
        gt_classes = np.zeros((nBoxes), np.int32) + self._class_to_index['particle']
        gt_overlaps = np.zeros((nBoxes, self._num_classes), np.float32)
        gt_overlaps[:, self._class_to_index['particle']] += 1.0
        return {'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': gt_overlaps, 'flipped': False}

    

