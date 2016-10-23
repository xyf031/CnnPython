

def resolve_caffe_log(log_name, get_data = False, to_be_print = False, write_to_file = True, write_to_pic = False):
  f_read = open(log_name, "r")
  f_strings = [i for i in f_read]
  f_read.close()

  patten1 = "Train net output #0:"
  patten2 = "Test net output #0:"
  train_loss = {}
  test_loss = {}
  test_accu = {}

  for ii in range(len(f_strings)):
    if patten1 in f_strings[ii]:
      tmp_train_iter = int(f_strings[ii - 1].split(" ")[5][0:-1])
      tmp_train_loss = float(f_strings[ii].split(" ")[14])
      train_loss[tmp_train_iter] = tmp_train_loss
      if to_be_print:
        print "Train", tmp_train_iter, ":\t", tmp_train_loss
      continue
    if patten2 in f_strings[ii]:
      tmp_test_iter = int(f_strings[ii - 1].split(" ")[5][0:-1])
      tmp_test_accu = float(f_strings[ii].split(" ")[14])
      tmp_test_loss = float(f_strings[ii + 1].split(" ")[14])
      test_loss[tmp_test_iter] = tmp_test_loss
      test_accu[tmp_test_iter] = tmp_test_accu
      if to_be_print:
        print "Test", tmp_test_iter, ":\t", tmp_test_accu, "\t", tmp_test_loss

  train_iters = train_loss.keys()
  test_iters = test_loss.keys()
  train_iters.sort()
  test_iters.sort()
  if write_to_file:
    f_write = open(log_name + "-Result-Data.txt", "w")
    for ii in train_iters:
      f_write.writelines("Train-Loss " + str(ii) + " : \t" + str(train_loss[ii]) + "\r\n")
    for ii in test_iters:
      f_write.writelines("Test-Loss-Accuracy " + str(ii) + " : \t" + str(test_loss[ii]) + " \t" + str(test_accu[ii]) + "\r\n")
    f_write.close()
    print log_name + "-Result-Data.txt --- has been writen."

  # if write_to_pic:
    # import matplotlib

  if get_data:
    return [train_loss, test_loss, test_accu]
