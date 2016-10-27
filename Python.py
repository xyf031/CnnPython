
import resolvelog
resolve_caffe_log(log_name, data = False, println = False, file = True, pic = False, pic_data = True)
return [train_iters, train_loss_list, test_iters, test_loss_list, test_accu_list]

import convertlmdb
print_labels(dir_path, print_cols = 100, file_out=False, return_labels=False)
return labels  # labels = []
generate_max_differ(dir_path)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6), dpi=300)
mypl = plt.subplot(111)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)

mypl.plot(xs, ys, 'r.-', linewidth=0.5, alpha=0.8, markersize=3)
mypl.plot(xs1, ys1, 'b.-', linewidth=0.5, alpha=0.8, markersize=3)
mypl.text(x, y, str, fontsize=5)

mypl.axis([x_min, x_max, y_min, y_max])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig(log_name + ".png", dpi=300)


# import readline, rlcompleter
# readline.parse_and_bind("tab: complete")  # for Linux
# readline.parse_and_bind("bind ^I rl_complete")  # for Mac
# Save this to ~/.pythonstartup
# and append to ~/.bashrc: export PYTHONSTARTUP=~/.pythonstartup

# import os
# cl='os.system("clear")'
# exec(cl)
# os.system("clear")
# os.chdir("/home/xyf/caffe-master/")

# import commands
# a = commands.getoutput('cat SF/maxdiffer_log.txt | grep "Train net output #0:"')


# import sys
# sys.path.insert(0, "/home/xyf/caffe-master/python")
# sys.path.insert(0, "/Users/x/Documents/caffe-master/python")

# vim ~/.bashrc
# export PYTHONPATH=~/caffe-master/python
# source ~/.bashrc


def pl(mylist):
  for ii in mylist:
    print ii


def pd(my_pic):
  for i_row in my_pic:
    for i_col in i_row:
      if i_col < 0.00001:
        print "    ",
      else:
        print "%.2f" % i_col,
    print " "


# from pylab import *
import numpy

import caffe
caffe.set_mode_gpu()


solver = caffe.SGDSolver("SF/lenetSF-solver.prototxt")
# pl(dir(solver))


# ---------- Train Net ----------
trainNet = solver.net

# ['data', 'label', 'conv1', 'pool1', 'conv2', 'pool2', 'ip1', 'ip2', 'loss']
trainNameBlobs = [i for i in solver.net._blob_names]
# ['mnist', 'conv1L', 'pool1L', 'conv2L', 'pool2L', 'ip1L', 'relu1L', 'ip2L', 'lossL'] -- Already Ordered.
trainNameLayers = [i for i in solver.net._layer_names]

# OrderedDict = {"BlobName": Blob Object}
trainBlobs = solver.net.blobs
tmpBlob = trainBlobs[trainNameBlobs[0]]
# blob.data blob.diff <--ndarray  blob.num blob.channels blob.height blob.width

# <caffe._caffe.LayerVec object at 0x34fabb0>  DIFFERENT from trainBlobs!
trainLayers = solver.net.layers
tmpLayer = trainLayers[0]
# layer.type = "Convolution" layer.reshape(..) layer.setup(..) layer.blobs <-- caffe._caffe.BlobVec object

# Train net parameters, include only LEARNABLE Layers!
# OrderedDict([('conv1L', <caffe._caffe.BlobVec object at 0x34fad70>), 
# ('conv2L', <caffe._caffe.BlobVec object at 0x34faa60>), 
# ('ip1L', <caffe._caffe.BlobVec object at 0x34fa590>), 
# ('ip2L', <caffe._caffe.BlobVec object at 0x34fa830>)])
trainParams = solver.net.params



# ---------- Test Net ----------
testNet = solver.test_nets[0]

testNameBlobs = [i for i in testNet._blob_names]
testNameLayers = [i for i in testNet._layer_names]

testBlobs = testNet.blobs
testLayers = testNet.layers

testEstLabels = solver.test_nets[0].blobs['ip2'].data.argmax(1)
testRealLabels = solver.test_nets[0].blobs['label'].data


# ---------- Record Train ----------
max_iter = 10000
trainLoss = numpy.zeros(max_iter)  # Vector: max_iter * 1
trainLoss[i] = solver.net.blobs['loss'].data  # Type: float, not list. The loss of this iteration.

solver.step(max_iter)
# for i in range(max_iter):
#   solver.step(1)
pd(solver.test_nets[0].blobs['data'].data[0][0])





# ---------------------------------------- Test Accuracy ----------------------------------------
tmpCorrect = 0
tmpLoss = 0
for i in range(test_iter):
  solver.test_nets[0].forward()

  # data.argmax(1) means find the max-arg in different pic. blobs["ip2"].shape = (batch-size, 10)
  tmpCorrect += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
  tmpLoss += solver.test_nets[0].blobs['loss'].data

testAccu[testI] = tmpCorrect * 0.01 / test_iter  # We have to multiply 0.01 because the test-batch size is 100.
testLoss[testI] = tmpLoss / test_iter  # We average between different batch, but we sum the loss within one batch.
# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print Wrong Pic ----------------------------------------
# ndarray, 100 * 1
testWrongIndex = solver.test_nets[0].blobs['ip2'].data.argmax(1) != solver.test_nets[0].blobs['label'].data
if len(testWrongIndex) == 0:
  continue

# ndarray, wrongCount * 10
testWrongIP2 = solver.test_nets[0].blobs['ip2'].data[testWrongIndex]
testWrongAns = solver.test_nets[0].blobs['label'].data[testWrongIndex]


# ndarray, wrongCount * 1 * 28 * 28
testWrongPics = solver.test_nets[0].blobs['data'].data[testWrongIndex]
# list of ndarray, wrongCount * 28 * 28
testWrongPicsInt = []
for i in testWrongPics:
  tmp_rows = numpy.zeros((solver.test_nets[0].blobs['data'].height, solver.test_nets[0].blobs['data'].width))
  for j in range(solver.test_nets[0].blobs['data'].height):
    for k in range(solver.test_nets[0].blobs['data'].width):
      tmp_rows[j][k] = int(i[0][j][k] * 255)
  testWrongPicsInt.append(tmp_rows)


import Image
imgOut = Image.fromarray(testWrongPicsInt[0])
imgOut = imgOut.convert("RGB")
imgOut.save("aaa.jpg")  # Ignore: "ValueError: ... Should be between 2 and 4"
# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print ----------------------------------------

# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print ----------------------------------------

# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print ----------------------------------------

# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print ----------------------------------------

# ---------------------------------------- *** ----------------------------------------



# ---------------------------------------- Print ----------------------------------------

# ---------------------------------------- *** ----------------------------------------




os.system("clear")
