
import numpy as np
import os
import scipy.sparse
import scipy.io as sio
# import cPickle

import datasets.imdb
import mrc_data

class mrc_data(datasets.imdb):
    def __init__(self, dataset_path):
        self._name = os.path.split(dataset_path)[1]  # Name of the protein.
        self._root_path = dataset_path  # Absolute path of the dataset folder.

        self._classes = ['__background__', 'particle', 'ice']
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
        mrc_txt_path = os.path.join(dataset_path, 'mrc.txt')
        star_txt_path = os.path.join(dataset_path, 'star.txt')
        mat_path = os.path.join(dataset_path, 'selective_search', '1.mat')
        if not (os.path.exists(mrc_txt_path) and os.path.exists(star_txt_path) and os.path.exists(mat_path)):
            print "ERROR: read_ALL() fails."
            return

        # Read mrc.txt
        with open(mrc_txt_path) as f:
            self._image_index = [i.strip() for i in f.readlines()]
        # Read star.txt
        with open(star_txt_path) as f:
            self._star_index = [i.strip() for i in f.readlines()]

        gt_roidb = self.read_all_star()
        ss_roidb = self.read_selective_search_mat(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        self._roidb = roidb

    def read_selective_search_mat(self, all_star):
        mat_path = os.path.join(self._root_path, 'selective_search', '1.mat')
        if not os.path.exists(mat_path):
            print "ERROE: ", mat_path, " doesn't exists. DEF read_selective_search_mat(gt_roidb) fails."
            return

        raw_data = sio.loadmat(mat_path)['boxes'].ravel()
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
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

    









def analyse_dataset(folder_path):
    '''
    ls folder_path/*.bmp
    ls folder_path/*.star
    Analyse the two results, and pair the .bmp file and .star file.
    Create mrc.txt, star.txt files, each contains the absolute path of bmp or star file.
    '''
    if not os.path.exists(folder_path):
        print "ERROR: ", folder_path, " doesn't exists. DEF analyse_dataset(folder_path) fails."
        # return [[], []]
        return
    abs_root = os.path.abspath(folder_path)

    import commands
    mrc_list = commands.getoutput('ls ' + os.path.join(abs_root, '*.bmp')).split()
    star_list = commands.getoutput('ls ' + os.path.join(abs_root, '*.star')).split()

    mrc_dict = {}
    star_dict = {}
    for ii in mrc_list:
        ii1 = os.path.split(ii)[1]
        mrc_dict[_mrc_filename_to_int(ii1)] = ii1
    for ii in star_list:
        ii1 = os.path.split(ii)[1]
        star_dict[_star_filename_to_int(ii1)] = ii1

    mrc_keys = set(mrc_dict.keys())
    star_keys = set(star_dict.keys())
    common_keys = mrc_keys & star_keys
    mrc_alone = mrc_keys - common_keys
    star_alone = star_keys - common_keys

    f_log = open(os.path.join(abs_root, 'analyse_log.txt'), 'w')
    if len(mrc_alone) > 0:
        f_log.writelines("MRC files not included: (cannot be resolved or cannot find .star file)\r\n")
        mrc_alone_sort = list(mrc_alone)
        mrc_alone_sort.sort()
        for ii in mrc_alone_sort:
            f_log.writelines(mrc_dict[ii] + '\r\n')
    if len(star_alone) > 0:
        f_log.writelines("\r\nSTAR files not included: (cannot be resolved or cannot find .mrc file)\r\n")
        star_alone_sort = list(star_alone)
        star_alone_sort.sort()
        for ii in star_alone_sort:
            f_log.writelines(star_dict[ii] + "\r\n")

    mrc_result = []
    star_result = []
    if len(common_keys) > 0:
        f_mrc = open(os.path.join(abs_root, 'mrc.txt'), 'w')
        f_star = open(os.path.join(abs_root, 'star.txt'), 'w')
        f_log.writelines("\r\n\r\nRelationship between *.mrc and *.star:\r\n")
        common_keys_sort = list(common_keys)
        common_keys_sort.sort()
        for ii in common_keys_sort:
            tmp1 = os.path.join(abs_root, mrc_dict[ii])
            tmp2 = os.path.join(abs_root, star_dict[ii])
            mrc_result.append(tmp1)
            star_result.append(tmp2)
            f_mrc.writelines(tmp1 + "\r\n")
            f_star.writelines(tmp2 + "\r\n")
            f_log.writelines(mrc_dict[ii] + " \t--->\t " + star_dict[ii] + '\r\n')
        f_mrc.close()
        f_star.close()
    
    f_log.writelines("Successfully paired: " + str(len(common_keys)) + " pairs.\r\n")
    f_log.writelines("MRC not included: " + str(len(mrc_alone)) + " mrc files.\r\n")
    f_log.writelines("STAR not included: " + str(len(star_alone)) + " star files.\r\n")
    f_log.close()
    # return [abs_root, mrc_result, star_result]


mrc_hard_count = 0
star_hard_count = -1000
def _mrc_filename_to_int(filename):
    try:
        intstr = filename[6:10]
        res = int(intstr)
    except:
        begin = -1
        end = -1
        for ii in range(len(filename)):
            if '0' <= filename[ii] <= '9' and begin < 0:
                begin = ii
                continue
            if '0' <= filename[ii] <= '9' and begin >= 0:
                end = ii
                break
        intstr = filename[begin:(end + 1)]
        try:
            res = int(intstr)
        except:
            mrc_data.mrc_hard_count -= 1
            return mrc_data.mrc_hard_count
    return res


def _star_filename_to_int(filename):
    try:
        intstr = filename[6:10]
        res = int(intstr)
    except:
        begin = -1
        end = -1
        for ii in range(len(filename)):
            if '0' <= filename[ii] <= '9' and begin < 0:
                begin = ii
                continue
            if '0' <= filename[ii] <= '9' and begin >= 0:
                end = ii
                break
        intstr = filename[begin:(end + 1)]
        try:
            res = int(intstr)
        except:
            mrc_data.star_hard_count -= 1
            return mrc_data.star_hard_count
    return res












