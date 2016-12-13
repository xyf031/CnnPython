
# Version 3.9
# Version: 2016-12-11-12:30
# By Xiao Yifan

import os
import numpy as np

import datasets.imdb

class mrc_data(datasets.imdb):

    def __init__(self, dataset_path):
        self._name = os.path.split(dataset_path)[1]  # Name of the protein.
        if len(self._name) == 0:
            self._name = os.path.split(os.path.split(dataset_path)[0])[1]
        self._root_path = dataset_path  # Absolute path of the dataset folder.

        self._classes = ['__background__', 'particle']
        self._num_classes = len(self._classes)
        self._index_to_class = dict(zip(xrange(self._num_classes), self._classes))
        self._class_to_index = dict(zip(self._classes, xrange(self._num_classes)))

        self._image_index = []  # List of *.bmp file absolute path.
        self._image_type = 'bmp'  # bmp.txt

        self._roidb = None
        self._roidb_handler = self.default_roidb

        self.config = {'cleanup'  : True,
                       'use_salt' : True}
                       #'top_k'    : 2000}

        self._box_side_length = 100
        self._read_ALL()  # <---------- Entrance


    def default_roidb(self, i):
        return 0

    def image_path_at(self, i):
        return self._image_index[i]

    def star_path_at(self, i):
        return self._star_index[i]

    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True


    def _read_ALL(self):
        self.read_ALL(self._root_path)

    def read_ALL(self, dataset_path):
        # Read: bmp.txt, *.gtroi, *.boxes

        bmp_txt_path = os.path.join(dataset_path, self._image_type + '.txt')  # bmp.txt
        if not os.path.exists(bmp_txt_path):
            print "ERROR: read_ALL() fails. bmp.txt do not exist."
            return

        # Read bmp.txt
        with open(bmp_txt_path) as f:
            self._image_index = [i.strip() for i in f.readlines()]

        # Read all *.gtroi
        gt_roidb = []
        for ii in self._image_index:
            fGt = open(ii[0:(-1 * len(self._image_type))] + "gtroi", "r")  # *.gtroi
            gtLines = [i.strip().split() for i in fGt.readlines()]
            fGt.close()

            nBoxes = len(gtLines)
            boxes = self._strMatrix2intMatrix(gtLines)
            gt_classes = np.zeros((nBoxes), np.int32) + self._class_to_index['particle']
            gt_overlaps = np.zeros((nBoxes, self._num_classes), np.float32)
            gt_overlaps[:, self._class_to_index['particle']] = 1.0
            gt_roidb.append({'boxes':boxes,'gt_classes':gt_classes,'gt_overlaps':gt_overlaps,'flipped':False})

        ss_roidb = self.read_all_boxes(gt_roidb)  # Read *.boxes
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        self._roidb = roidb


    def read_all_boxes(self, all_star):
        # Read all *.boxes
        # The *.boxes must have same name with *.bmp

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


    def evaluate_detections(self, all_boxes, nms=0.9, imgavg=300, imgmax=600):
        # Generate *.pick, *-pick.jpg, Res-time.txt

        # Generate *.pick
        self._write_results_file(all_boxes)

        # Analyse *.pick and *.gtroi
        TruePositive = 0
        FalsePositive = 0
        FalseNegative = 0
        DistanceSum = 0  # Only TruePositive included.
        IoUSum = 0  # Only TruePositive included.
        mAP = []  # len() = len(self._image_index)

        for ii in self._image_index:
            pick_path = ii[0:(-1 * len(self._image_type))] + 'pick'
            gtroi_path = ii[0:(-1 * len(self._image_type))] + 'gtroi'

            # Generate *-pick.jpg
            self._draw_pick_gtroi(ii, pick_path, gtroi_path)
            
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

        fResultName = 'Res-NMS=' + str(nms) + '-AVG' + str(imgavg) + '-MAX' + str(imgmax) + '.txt'
        fResult = open(os.path.join(self._root_path, fResultName), 'w+')
        fResult.writelines('\r\nTruePositive:\r\n' + str(TruePositive) + '\r\n')
        fResult.writelines('\r\nFalsePositive:\r\n' + str(FalsePositive) + '\r\n')
        fResult.writelines('\r\nFalseNegative:\r\n' + str(FalseNegative) + '\r\n')
        fResult.writelines('\r\nRecall:\r\n' + str(Recall) + '\r\n')
        fResult.writelines('\r\nPrecision:\r\n' + str(Precision) + '\r\n')
        fResult.writelines('\r\nAverage Distance of TP-rois(Lower the Better)\r\n' + str(AverageDist) + '\r\n')
        fResult.writelines('\r\nAverage IoU of TP-rois(Higher the Better)\r\n' + str(AverageIoU) + '\r\n')
        fResult.writelines('\r\nAverage mAP among each image(Higher the Better)\r\n' + str(AveragemAP) + '\r\n')
        fResult.close()
        print "\nRecall = " + str(Recall)
        print "TP = " + str(TruePositive)
        print "FP = " + str(FalsePositive)
        print "FN = " + str(FalseNegative)
        print "Precision = " + str(Precision)
        print "Dist = " + str(AverageDist)
        print "IoU = " + str(AverageIoU)
        print "mAP = " + str(AveragemAP)

        resultPath = os.path.join(self._root_path, fResultName[0:-4])
        if os.path.exists(resultPath):
            print "Python WARNING:  Results not moved!"
        else:
            os.mkdir(resultPath)
            os.system('mv ' + os.path.join(self._root_path, '*.jpg') + ' ' + resultPath)
            os.system('mv ' + os.path.join(self._root_path, '*.pick') + ' ' + resultPath)
            os.system('mv ' + os.path.join(self._root_path, fResultName) + ' ' + resultPath)
            

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

    def _draw_pick_gtroi(self, image_path, pick_path, gtroi_path):
        # Draw *.pick and *.gtroi on the *.bmp  ---> *-pick.jpg

        if not (os.path.exists(pick_path) and os.path.exists(gtroi_path)):
            print "ERROR: _draw_pick_gtroi() fails. *.pick or *.gtroi do not exist."
            return
        fPick = open(pick_path, 'r')
        fGtroi = open(gtroi_path, 'r')
        pick_list = [ii.strip().split() for ii in fPick.readlines()]
        gt_list = [ii.strip().split() for ii in fGtroi.readlines()]
        fPick.close()
        fGtroi.close()

        import cv2
        im = cv2.imread(image_path)
        for ii in pick_list:
            cv2.rectangle(im, (int(ii[0]), int(ii[1])), (int(ii[2]), int(ii[3])), (0, 0, 255), thickness=7)  # Red
        for ii in gt_list:
            cv2.rectangle(im, (int(ii[0]), int(ii[1])), (int(ii[2]), int(ii[3])), (255, 0, 0), thickness=5)  #Blue
        cv2.imwrite(image_path[0:(-1 * len(self._image_type) - 1)] + '-pick.jpg', im)  # *-pick.jpg


    def Pair_Pick_Gtroi(self, pick_path, gtroi_path):
        # Pair *.pick and *.gtroi
        # return {TP, FP, FN, Dist, IoU, mAP}
        # {'TP': int, 'FP': int, 'FN': int, 'Dist': float, 'IoU': float, 'mAP': float}
        IOU_THRES = 0.65

        if not (os.path.exists(pick_path) and os.path.exists(gtroi_path)):
            print "ERROR: Pair_Pick_Gtroi() fails. *.pick or *.gtroi do not exist."
            print "return \{\}: " + pick_path
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
            print "return \{\}: " + pick_path
            return {}
        picknumpy = self._strMatrix2intMatrix(pick_list)  # pick_list is R * 5, but picknumpy is R*4
        gtnumpy = self._strMatrix2intMatrix(gt_list)  # G * 4

        if self._box_side_length != (picknumpy[0][2] - picknumpy[0][0]):
            print "ERROR: Pair_Pick_Gtroi(). Side-Length in *.pick in wrong!"
            print pick_path
            # return {}
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






        