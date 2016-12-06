
import os
import commands
import mrc_data_global

def analyse_dataset(folder_path, file_type = "bmp"):
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

    mrc_list = commands.getoutput('ls ' + os.path.join(abs_root, '*.' + file_type)).split()
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
        f_mrc = open(os.path.join(abs_root, file_type + '.txt'), 'w')
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
            mrc_data_global.mrc_hard_count -= 1
            return mrc_data_global.mrc_hard_count
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
            mrc_data_global.star_hard_count -= 1
            return mrc_data_global.star_hard_count
    return res


def convert2bmp(root_path):
    # Convert *.mrc.bmp to *.bmp
    mrcbmp_list = commands.getoutput('ls ' + os.path.join(root_path, '*.mrc.bmp')).split()
    bmp_list = []
    for ii in mrcbmp_list:
        bmp_list.append(ii[0:-8] + '.bmp')
    for ii in range(0, len(bmp_list)):
        os.system('mv ' + mrcbmp_list[ii] + ' ' + bmp_list[ii])


import sys, getopt
opts, args = getopt.getopt(sys.argv[1:], "hf:t:", ["version", "file="])
# print opts  # Defined args
# print args  # Undefined args
# print sys.argv  # Origional args

folder_path = sys.argv[1]
if len(sys.argv) >= 3:
    file_type = sys.argv[2]
else:
    file_type = ""

for op, value in opts:
    if op == "-f":
        folder_path = value
    elif op == "-t":
        file_type = value
    elif op == "-h":
        print "python analyse_dataset.py /home/xyf/ssd/gammas mrc"
        print "python analyse_dataset.py -f /home/xyf/ssd/gammas -t bmp"
        print "python analyse_dataset.py -h"
        print "You need copy mrc_data_global.py to your python path."
        sys.exit()

if len(file_type) > 0:
    analyse_dataset(folder_path, file_type)
else:
    analyse_dataset(folder_path)

