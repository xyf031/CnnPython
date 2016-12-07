
# Version 3.5
# Version: 2016-12-07-20:30
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

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True



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
            fBox = open(i[0:(-1 * len(self._image_type))] + "boxes", "r")
            boxLines = [ii.strip().split() for ii in fBox.readlines()]
            all_boxes.append(self._strMatrix2intMatrix(boxLines))
        return self.create_roidb_from_box_list(all_boxes, all_star)

    def _strMatrix2intMatrix(self, strMatrix):
        results = np.zeros((len(strMatrix), 4), dtype=np.int32)
        for i in range(0, len(strMatrix)):
            ii = strMatrix[i]
            results[i] = [int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3])]
        return results



    def evaluate_detections(self, all_boxes, output_dir=None):
        # Generate *.pick and *.pick-out
        self._write_results_file(all_boxes)

        TruePositive = 0
        FalsePositive = 0
        FalseNegative = 0
        DistanceSum = 0  # Only TruePositive included.
        IoUSum = 0  # Only TruePositive included.
        mAP = []  # len() = len(self._image_index)
        for ii in self._image_index:
            pick_path = ii[0:(-1 * len(self._image_type))] + 'pick'
            gtroi_path = ii[0:(-1 * len(self._image_type))] + 'gtroi'
            self._draw_pick(ii, pick_path)
            
            # Pair *.pick and *.gtroi
            ii_eval = self.Pair_Pick_Gtroi(pick_path, gtroi_path)
            if len(ii_eval) == 0:
                continue
            TruePositive += ii_eval['TP']
            FalsePositive += ii_eval['FP']
            FalseNegative += ii_eval['FN']
            DistanceSum += ii_eval['Dist']
            IoUSum += ii_eval['IoU']
            mAP.append(ii_eval['mAP'])
        Recall = (1.0 * TruePositive) / (TruePositive + FalseNegative)
        Precision = (1.0 * TruePositive) / (TruePositive + FalsePositive)
        AverageDist = np.sqrt((1.0 * DistanceSum) / TruePositive)
        AverageIoU = (1.0 * IoUSum) / TruePositive
        AveragemAP = np.mean(mAP)

        import time
        t1 = time.ctime()
        t2 = t1.replace(':', '-')
        t3 = t2.replace(' ', '_')
        fResult = open('Res-' + t3 + '.txt', 'w+')
        fResult.writelines('\r\nTruePositive:\r\n' + str(TruePositive) + '\r\n')
        fResult.writelines('\r\nFalsePositive:\r\n' + str(FalsePositive) + '\r\n')
        fResult.writelines('\r\nFalseNegative:\r\n' + str(FalseNegative) + '\r\n')
        fResult.writelines('\r\nRecall:\r\n' + str(Recall) + '\r\n')
        fResult.writelines('\r\nPrecision:\r\n' + str(Precision) + '\r\n')
        fResult.writelines('\r\nAverage Distance of TP-rois(Lower the Better)\r\n' + str(AverageDist) + '\r\n')
        fResult.writelines('\r\nAverage IoU of TP-rois(Higher the Better)\r\n' + str(AverageIoU) + '\r\n')
        fResult.writelines('\r\nAverage mAP among each image(Higher the Better)\r\n' + str(AveragemAP) + '\r\n')
        fResult.close()

    def _write_results_file(self, all_boxes):
        # all_boxes[class_id][image_id] = N x 5 array of detections in (x1, y1, x2, y2, score)

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
            f = open(filename + '-out', 'wt')
            if otherRoi == []:
                f.write('\r\n')
            else:
                rois = otherRoi
                for k in xrange(rois.shape[0]):
                    f.write('{:d} {:d} {:d} {:d} \t{:.3f}\r\n'.
                        format(int(rois[k, 0]), int(rois[k, 1]), int(rois[k, 2]), int(rois[k, 3]), rois[k, -1]))
            f.close()

    def _draw_pick(self, image_path, pick_path):
        if not os.path.exists(pick_path):
            print "ERROR: _draw_pick() fails. *.pick do not exist."
            return
        import cv2
        im = cv2.imread(image_path)
        fPick = open(pick_path, 'r')
        pick_list = [ii.strip().split() for ii in fPick.readlines()]
        fPick.close()
        for ii in pick_list:
            cv2.rectangle(im, (int(ii[0]), int(ii[1])), (int(ii[2]), int(ii[3])), (0, 0, 255), thickness=7)  # Red
        cv2.imwrite(image_path + '.jpg', im)

    def Pair_Pick_Gtroi(self, pick_path, gtroi_path):
        # Pair *.pick and *.gtroi
        # return {TP, FP, FN, Dist, IoU, mAP}
        # {'TP': int, 'FP': int, 'FN': int, 'Dist': float, 'IoU': float, 'mAP': float}
        IOU_THRES = 0.8

        if not (os.path.exists(pick_path) and os.path.exists(gtroi_path)):
            print "ERROR: Pair_Pick_Gtroi() fails. *.pick or *.gtroi do not exist."
            return {}
        fPick = open(pick_path, 'r')
        fGtroi = open(gtroi_path, 'r')
        pick_list = [ii.strip().split() for ii in fPick.readlines()]
        gt_list = [ii.strip().split() for ii in fGtroi.readlines()]
        fPick.close()
        fGtroi.close()

        summaryPickN = len(pick_list)
        summaryGtN = len(gt_list)
        if summaryGtN <= 0 or summaryPickN <= 0:
            print "SKIP: %s or %s has nothing!" % (pick_path, gtroi_path)
            return {}
        picknumpy = self._strMatrix2intMatrix(pick_list)  # pick_list is R * 5, but picknumpy is R*4
        gtnumpy = self._strMatrix2intMatrix(gt_list)  # G * 4

        if self._box_side_length != (pick_list[0][2] - pick_list[0][0]):
            print "ERROR: _pair_pick_gtroi(). Side-Length in *.pick in wrong!"
            return {}
        SIDE = self._box_side_length

        IoU0 = np.zeros((summaryPickN, summaryGtN))
        IoU1 = np.ones((summaryPickN, summaryGtN))
        # pickX1, pickY1
        interXp = np.dot(picknumpy[:, 0].reshape((summaryPickN, 1)), np.ones((1, summaryGtN)))
        interYp = np.dot(picknumpy[:, 1].reshape((summaryPickN, 1)), np.ones((1, summaryGtN)))
        # gtroiX1, gtroiY1
        interXg = np.dot(np.ones((summaryPickN, 1)), gtnumpy[:, 0].reshape((1, summaryGtN)))
        interYg = np.dot(np.ones((summaryPickN, 1)), gtnumpy[:, 1].reshape((1, summaryGtN)))
        # max(0, SIDE - abs(x1 - X1))
        interXlen = SIDE*IoU1 - np.abs(interXp - interXg)
        interXlen[interXlen < 0] = 0
        interYlen = SIDE*IoU1 - np.abs(interYp - interYg)
        interYlen[interYlen < 0] = 0
        # IoU, dtype=np.float64
        IoU = (interXlen * interYlen) / (2*SIDE*SIDE*IoU1 - interXlen * interYlen)

        # IoU[i, pickMaxId[i]] == pickMaxIoU[i]
        pickMaxIoU = np.amax(IoU, 1)
        pickMaxId = np.argmax(IoU, 1)
        pickPaired = (pickMaxIoU >= IOU_THRES)
        gtMaxIoU = np.amax(IoU, 0)
        gtMaxId = np.argmax(IoU, 0)
        gtPaired = (gtMaxIoU >= IOU_THRES)
        summaryTP = np.sum(pickPaired)  # Always == np.sum(gtPaired)
        summaryFP = summaryPickN - summaryTP
        summaryFN = summaryGtN - np.sum(gtPaired)

        # DistanceSquaredSum, IoUSum among TruePositive
        summaryIoUSum = 0.0  # Higher the better.
        summaryDistance = 0.0  # Lower the better.
        for ii in range(0, summaryPickN):
            if pickPaired[ii]:
                summaryIoUSum += pickMaxIoU[ii]
                summaryDistance += (picknumpy[ii, 0] - gtnumpy[pickMaxId[ii], 0])**2
                summaryDistance += (picknumpy[ii, 1] - gtnumpy[pickMaxId[ii], 1])**2

        # mAP
        pickScores = np.zeros(summaryPickN)
        for ii in range(0, summaryPickN):
            pickScores[ii] = float(pick_list[ii][-1])
        pickSort = np.argsort(pickScores)
        summarymAP = 0.0
        trueCount = 0.0
        for ii in range(0, summaryPickN):
            iii = summaryPickN - 1 - ii
            if pickPaired[pickSort[iii]]:
                trueCount += 1.0
                summarymAP += trueCount / (ii + 1)
        summarymAP = summarymAP / trueCount

        return {'TP': summaryTP, 'FP': summaryFP, 'FN': summaryFN, 'Dist': summaryDistance, 
        'IoU': summaryIoUSum, 'mAP': summarymAP}






        