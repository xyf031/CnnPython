

def resolve_caffe_log(dir_path):
  f_read = open(dir_path, "r")
  f_strings = [i for i in f_read]
  f_read.close()

  patten1 = "Train net output #0:"
  patten2 = "Test net output #0:"

  for ii in range(len(f_strings)):
    if patten1 in f_strings[ii]:
      train_iter = int(f_strings[ii - 1].split(" ")[5][0:-1])
      train_loss = float(f_strings[ii].split(" ")[14])
      # print train_iter, ":", train_loss
      continue
    if patten2 in f_strings[ii]:
      test_iter = int(f_strings[ii - 1].split(" ")[5][0:-1])
      test_accu = float(f_strings[ii].split(" ")[14])
      test_loss = float(f_strings[ii + 1].split(" ")[14])
      # print test_iter, ":", test_accu, test_loss



