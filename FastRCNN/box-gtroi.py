

import os
import commands
import numpy as np

def evaluate_detections(root_path):
    # Generate Res-time.txt

    boxes_list = commands.getoutput('ls ' + os.path.join(root_path, '*.boxes')).split()
    gt_list = commands.getoutput('ls ' + os.path.join(root_path, '*.gtroi')).split()
    if not len(boxes_list) == len(gt_list):
        print "ERROR: fileNum.  len(boxes) = " + str(len(boxes_list)) + ' \tlen(gtroi)' + str(len(gt_list))
        return

    # Analyse *.boxes and *.gtroi
    TruePositive = 0
    FalsePositive = 0
    FalseNegative = 0
    DistanceSum = 0  # Only TruePositive included.
    IoUSum = 0  # Only TruePositive included.
    mAP = []  # len() = len(self._image_index)

    for ii in range(0, len(boxes_list)):
        if not boxes_list[ii][0:-5] == gt_list[ii][0:-5]:
            print "ERROR: different names.  " + boxes_list[ii] + ' \t' + gt_list[ii]
            continue

        ii_eval = Pair_Pick_Gtroi(boxes_list[ii], gt_list[ii])
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
    fResult = open(os.path.join(root_path, 'BoxesRecall-' + t3 + '.txt'), 'w+')
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


def _strMatrix2intMatrix(strMatrix):
    results = np.zeros((len(strMatrix), 4), dtype=np.int32)
    for i in range(0, len(strMatrix)):
        ii = strMatrix[i]
        results[i] = [int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3])]
    return results


def Pair_Pick_Gtroi(pick_path, gtroi_path):
    # Pair *.pick and *.gtroi
    # return {TP, FP, FN, Dist, IoU, mAP}
    # {'TP': int, 'FP': int, 'FN': int, 'Dist': float, 'IoU': float, 'mAP': float}
    IOU_THRES = 0.65
    print "Reading:-----" + pick_path + '\t IoU_Thresh = ' + str(IOU_THRES)

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
    picknumpy = _strMatrix2intMatrix(pick_list)  # pick_list is R * 5, but picknumpy is R*4
    gtnumpy = _strMatrix2intMatrix(gt_list)  # G * 4

    if 100 != (picknumpy[0][2] - picknumpy[0][0]):
        print "ERROR: _pair_pick_gtroi(). Side-Length in *.pick in wrong!"
        return {}
    SIDE = 100

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



if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        evaluate_detections('./')
    else:
        evaluate_detections(sys.argv[1])
















