
# Version 3.0
# Version: 2016-12-06-14:00
# By Xiao Yifan

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
        # self._classes = ['__background__', 'particle', '2', '3', '4', '5', '6', '7', 
        # '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        self._num_classes = len(self._classes)
        self._index_to_class = dict(zip(xrange(self._num_classes), self._classes))
        self._class_to_index = dict(zip(self._classes, xrange(self._num_classes)))

        self._image_index = []  # List of *.bmp file absolute path.
        self._star_index = []  # List of *.star file absolute path.

        self._image_type = 'bmp'  # bmp.txt
        self._roidb = None
        self._roidb_handler = self.default_roidb

        self.config = {'cleanup'  : True,
                       'use_salt' : True}
                       #'top_k'    : 2000}
        # self._obj_proposer = 'selective_search'
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
        comp_id = self._write_results_file(all_boxes)
        return

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _write_results_file(self, all_boxes):
        for im_ind, im_path in enumerate(self.image_index):
            filename = im_path[0:(-1 * len(self._image_type))] + "pick"  # *.pick
            rois = all_boxes[self._class_to_index['particle']][im_ind]
            f = open(filename, 'wt')
            if rois == []:
                f.write('\r\n')
            else:
                for k in xrange(rois.shape[0]):
                    f.write('{:d} {:d} {:d} {:d} \t{:.3f}\r\n'.
                        format(int(rois[k, 0]), int(rois[k, 1]), int(rois[k, 2]), int(rois[k, 3]), rois[k, -1]))
            f.close()

            otherRoi = all_boxes[0][im_ind]
            f = open(filename + '.out', 'wt')
            if otherRoi == []:
                f.write('\r\n')
            else:
                rois = otherRoi
                for k in xrange(rois.shape[0]):
                    f.write('{:d} {:d} {:d} {:d} \t{:.3f}\r\n'.
                        format(int(rois[k, 0]), int(rois[k, 1]), int(rois[k, 2]), int(rois[k, 3]), rois[k, -1]))
            f.close()

        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())
        return comp_id


    def _read_ALL(self):
        self.read_ALL(self._root_path)

    def read_ALL(self, dataset_path):
        bmp_txt_path = os.path.join(dataset_path, self._image_type + '.txt')  # bmp.txt
        star_txt_path = os.path.join(dataset_path, 'star.txt')
        # mat_path = os.path.join(dataset_path, 'selective_search.mat')
        if not (os.path.exists(bmp_txt_path) and os.path.exists(star_txt_path)):
            print "ERROR: read_ALL() fails. Some files do not exist."
            return

        # Read bmp.txt
        with open(bmp_txt_path) as f:
            self._image_index = [i.strip() for i in f.readlines()]
        # Read star.txt
        with open(star_txt_path) as f:
            self._star_index = [i.strip() for i in f.readlines()]

        gt_roidb = [self.read_star_file(star_path) for star_path in self._star_index]  # Read *.star
        ss_roidb = self.read_all_boxes(gt_roidb)  # Read *.boxes
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        self._roidb = roidb

        # Read selective_search.mat
        # ss_roidb = self.read_selective_search_mat(gt_roidb, mat_path)

    # def read_selective_search_mat(self, all_star, mat_path):
    #     # Read selective_search.mat
    #     raw_data = sio.loadmat(mat_path)['boxes'].ravel()
    #     box_list = []
    #     for i in xrange(raw_data.shape[0]):
    #         box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
    #         # Each item in raw_data is still a list, corresponds the Rois of the i-th pic.
    #         # Size of raw_data[i] is (roiNum, 4).
    #         # The columns of raw_data[i][j] is [rowBegin, colBegin, rowEnd, colEnd] in Matlab matrix.
    #         # In python indices, that means [yBegin, xBegin, yEnd, xEnd] + 1. (Matlab indices begin from 1).
    #     return self.create_roidb_from_box_list(box_list, all_star)

    def read_star_file(self, file_path):
        # Read all *.star
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
                # print ii
                continue
            else:
                xmin = max(0, x_center - self._box_side_length / 2)
                xmax = x_center + self._box_side_length / 2
                ymin = max(0, y_center - self._box_side_length / 2)
                ymax = y_center + self._box_side_length / 2
                boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    
        nBoxes = len(boxes)
        boxes = np.array(boxes, np.int32)
        gt_classes = np.zeros((nBoxes), np.int32) + self._class_to_index['particle']
        gt_overlaps = np.zeros((nBoxes, self._num_classes), np.float32)
        gt_overlaps[:, self._class_to_index['particle']] += 1.0
        return {'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': gt_overlaps, 'flipped': False}

    def read_all_boxes(self, all_star):
        # Read all *.boxes
        # The *.boxes must have same name with *.mrc/*.bmp

        all_boxes = []
        for i in self._image_index:
            fBox = open(i[0:-4] + ".boxes", "r")
            boxLines = [ii.strip().split() for ii in fBox.readlines()]
            all_boxes.append(self._strMatrix2intMatrix(boxLines))
        return self.create_roidb_from_box_list(all_boxes, all_star)

    def _strMatrix2intMatrix(self, strMatrix):
        results = np.zeros((len(strMatrix), 4), dtype=np.int32)
        for i in range(0, len(strMatrix)):
            ii = strMatrix[i]
            results[i] = [int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3])]
        return results




        