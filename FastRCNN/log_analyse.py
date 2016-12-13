
import os
import matplotlib.pyplot as plt

def getOutput1(log_file):
    f_log = open(log_file, 'r')
    log_list = [i.strip() for i in f_log.readlines()]
    f_log.close()

    learning_rate = 0.0001
    gamma = 0.1
    step_size = 10000

    iters = []
    class_los = []
    for i in range(0, len(log_list)):
        if 'solver.cpp:189] Iteration' in log_list[i]:
            tmp = log_list[i]
            try:
                iters.append(int(tmp.split()[5][0:-1]))
            except:
                print tmp
                continue

            j = 2
            while 'solver.cpp:204]     Train net output #1: loss_cls' not in log_list[i + j]:
                j += 1
            tmp1 = log_list[i + j]
            try:
                class_los.append(float(tmp1.split()[10]))
            except:
                class_los.append(0)
                print '######\t' + tmp1
            # i += (j + 1)

    input_path = os.path.split(log_file)
    output_path = os.path.join(input_path[0], 'Analyse-' + input_path[1])
    output_path_pic = os.path.join(input_path[0], 'Draw-' + input_path[1] + '.png')
    f_out = open(output_path, 'w')
    for i in range(0, len(iters)):
        f_out.writelines('Iters = \t' + str(iters[i]) + ' \tclass_los = \t' + str(class_los[i]) + '\r\n')
    f_out.close()

    myplt = plt.subplot(111)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
    if step_size < iters[-1]:
        lr_x = range(step_size, iters[-1], step_size)
        for i in lr_x:
            myplt.plot([i, i], [0.01, 0.8], 'r-', linewidth=0.5)
    myplt.plot(iters, class_los, '.-', linewidth=0.1, markersize=0.2)
    plt.savefig(output_path_pic, dpi=300)
    print output_path + ' \thas been generated.'
    print output_path_pic + ' \thas been generated.'



if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print 'python log_analyse.py /home/u/train_caffe_log.txt'
        exit(1)
    if not os.path.exists(sys.argv[1]):
        print 'No log.txt file.'
        exit(1)

    getOutput1(sys.argv[1])

