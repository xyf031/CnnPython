

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  mydict = cPickle.load(fo)
  fo.close()
  return mydict


dirPath = "data/cifar10/cifar10-py/"
fileList = ["data_batch_1", "data_batch_2", "data_batch_3", 
"data_batch_4", "data_batch_5", "test_batch", "batches.meta"]
from PIL import Image

I = 1
batchData = unpickle(dirPath + fileList[I])
batchContents = batchData.keys()
pics = batchData[batchContents[0]]
labels = batchData[batchContents[1]]
names = batchData[batchContents[3]]
for i in range(len(labels)):
  tmpResize = pics[i]
  tmpResize.resize((3, 32, 32))
  tmpResize = tmpResize.transpose((1, 2, 0))
  tmpImg = Image.fromarray(tmpResize, "RGB")
  tmpImg.save(dirPath + "pic" + str(I) + "/" + str(labels[i]) + "-" + '{:0>8d}'.format(i) + "-" + names[i][0:5] + ".bmp")

