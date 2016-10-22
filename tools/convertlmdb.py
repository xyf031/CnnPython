
import lmdb
from caffe.proto import caffe_pb2


def print_labels(dir_path, print_cols = 100, file_out=False, return_labels=False):
  lmdb_dir = lmdb.open(dir_path)
  lmdb_cursor = lmdb_dir.begin().cursor()
  caffe_datum = caffe_pb2.Datum()
  labels = []

  for id_lmdb, str_lmdb in lmdb_cursor:
    caffe_datum.ParseFromString(str_lmdb)
    labels.append(int(caffe_datum.label))
  rows = int(len(labels) / print_cols)
  for ii in range(rows):
    for jj in range(print_cols):
      print labels[ii * print_cols + jj],
    print " "
  for ii in range(len(labels) - rows * print_cols):
    print labels[rows * print_cols + ii],
  print " "

  if file_out:
    f_out = open(dir_path + "_labels.txt", "w")
    for ii in labels:
      f_out.writelines(str(ii) + "\r\n")
    f_out.close()
    print dir_path + "_labels.txt have been writen."
  if return_labels:
    return labels
  print "Done."


def generate_max_differ(dir_path):
  lmdb_in_dir = lmdb.open(dir_path)
  lmdb_cursor = lmdb_in_dir.begin().cursor()
  caffe_datum = caffe_pb2.Datum()
  data_in_labels = {}

  for id_lmdb, str_lmdb in lmdb_cursor:
    caffe_datum.ParseFromString(str_lmdb)
    label = int(caffe_datum.label)
    array = caffe.io.datum_to_array(caffe_datum)
    if label in data_in_labels:
      data_in_labels[label].append(array)
    else:
      data_in_labels[label] = [array]

  all_label = data_in_labels.keys()
  all_label.sort()
  all_label_length = {}
  all_data_size = 0
  for ii in all_label:
    all_label_length[ii] = len(data_in_labels[ii])
    all_data_size += len(data_in_labels[ii])

  lmdb_out_dir = lmdb.open(dir_path + "_max_differ", map_size=int(1e12))
  lmdb_writer = lmdb_out_dir.begin(write=True)
  ID = 0
  while (ID < all_data_size):
    for ii in all_label:
      if all_label_length[ii] > 0:
        caffe_datum = caffe.io.array_to_datum(data_in_labels[ii][all_label_length[ii] - 1], ii)
        IDstr = '{:0>8d}'.format(ID)
        lmdb_writer.put(IDstr, caffe_datum.SerializeToString())
        all_label_length[ii] -= 1
        ID += 1
  lmdb_writer.commit()






dir_path = "data/mnist/mnist_train_lmdb"
