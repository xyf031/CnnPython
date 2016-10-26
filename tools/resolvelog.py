

def resolve_caffe_log(log_name, data = False, println = True, file = True, pic = False, pic_data = True):
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
      tmp_train_loss = float(f_strings[ii].split()[10])
      for offset in range(5):
        if "Iteration " in f_strings[ii - 1 - offset] and "loss = " in f_strings[ii - 1 - offset]:
          tmp_train_iter = int(f_strings[ii - 1 - offset].split()[5][0:-1])
          break
      train_loss[tmp_train_iter] = tmp_train_loss
      if println:
        print "Train", tmp_train_iter, ":\t", tmp_train_loss
      continue

    if patten2 in f_strings[ii]:
      tmp_test_accu = float(f_strings[ii].split()[10])
      for offset in range(5):
        if "Iteration " in f_strings[ii - 1 - offset] and "Testing net " in f_strings[ii - 1 - offset]:
          tmp_test_iter = int(f_strings[ii - 1 - offset].split()[5][0:-1])
          break
      for offset in range(5):
        if "Test net output #1: loss = " in f_strings[ii + 1 + offset]:
          tmp_test_loss = float(f_strings[ii + 1].split()[10])
          break
      test_loss[tmp_test_iter] = tmp_test_loss
      test_accu[tmp_test_iter] = tmp_test_accu
      if println:
        print "Test", tmp_test_iter, ":\t", tmp_test_loss, "\t", tmp_test_accu

  train_iters = train_loss.keys()
  test_iters = test_loss.keys()
  train_iters.sort()
  test_iters.sort()
  train_loss_list = []
  test_loss_list = []
  test_accu_list = []
  for ii in train_iters:
    train_loss_list.append(train_loss[ii])
  for ii in test_iters:
    test_loss_list.append(test_loss[ii])
    test_accu_list.append(test_accu[ii])

  if file:
    f_write = open(log_name + "-Result-Data.txt", "w")
    for ii in train_iters:
      f_write.writelines("Train-Loss " + str(ii) + " : \t" + str(train_loss[ii]) + "\r\n")
    for ii in test_iters:
      f_write.writelines("Test-Loss-Accuracy " + str(ii) + " : \t" + str(test_loss[ii]) + " \t" + str(test_accu[ii]) + "\r\n")
    f_write.close()
    print log_name + "-Result-Data.txt --- has been writen."

  if pic:
    import matplotlib.pyplot as plt
    DPI = 400
    LOSS_LINE_WIDTH = 0.5
    LOSS_ALPHA = 0.8
    TEXT_LINE_WIDTH = 0.3
    TEXT_LINE_ALPHA = 0.2

    x_min = 0
    x_max = train_iters[-1] + train_iters[1] - train_iters[0]
    y_min = 0
    # y_max = train_loss_list[int((1 - 0.98) * len(train_loss_list))]
    # y_max = (train_loss_list[0] + train_loss_list[1] + train_loss_list[2]) / 3
    y_max = sum(train_loss_list[0:3]) / 3.0
    
    plt.figure(figsize=(10, 6), dpi=DPI)
    mypl = plt.subplot(111)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
    mypl.plot(train_iters, train_loss_list, 'r.-', linewidth=LOSS_LINE_WIDTH, alpha=LOSS_ALPHA, markersize=3)
    mypl.plot(test_iters, test_loss_list, 'b.-', linewidth=LOSS_LINE_WIDTH, alpha=LOSS_ALPHA, markersize=3)
    mypl.axis([x_min, x_max, y_min, y_max])

    if pic_data:
      y_skip = (y_max - y_min) / 50
      train_print_count = 0
      for ii in train_iters:
        tmp_y = y_min + 0.3 * (y_max - y_min) + y_skip * (train_print_count % 10)
        mypl.text(ii, tmp_y, str(train_loss_list[train_print_count]), fontsize=5)
        mypl.plot([ii, ii], [train_loss_list[train_print_count], tmp_y], 'k-', linewidth=TEXT_LINE_WIDTH, alpha=TEXT_LINE_ALPHA)
        train_print_count += 1
      
      test_print_count = 0
      for ii in test_iters:
        tmp_y = y_min + 0.7 * (y_max - y_min) + y_skip * (test_print_count % 10)
        mypl.text(ii, tmp_y, str(test_loss_list[test_print_count]), fontsize=5)
        mypl.plot([ii, ii], [test_loss_list[test_print_count], tmp_y], 'b-', linewidth=TEXT_LINE_WIDTH, alpha=TEXT_LINE_ALPHA)
        test_print_count += 1
    
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.savefig(log_name + ".png", dpi=DPI)
    print log_name + ".png --- has been created."

  if data:
    return [train_iters, train_loss_list, test_iters, test_loss_list, test_accu_list]


def compare_logs(iters1, list1, iters2, list2, out_name = "compareLogs.png"):
  length_of_data = len(iters1)
  if len(list1) != length_of_data or len(iters2) != length_of_data or len(list2) != length_of_data:
    print "compare_logs(): input length error!"
    return

  for ii in range(length_of_data):
    if iters1[ii] != iters2[ii]:
      print "compare_logs(): iters not equal"
      return

  import matplotlib.pyplot as plt
  DPI = 400
  LOSS_LINE_WIDTH = 0.5
  LOSS_ALPHA = 0.8
  x_min = 0
  x_max = iters1[-1] + iters1[1] - iters1[0]
  y_min = 0
  y_max = max(sum(list1[0:3]) / 3.0, 1.0)
  # y_max = 1

  plt.figure(figsize=(10, 6), dpi=DPI)
  mypl = plt.subplot(111)
  plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
  mypl.plot(iters1, list1, 'r.-', linewidth=LOSS_LINE_WIDTH, alpha=LOSS_ALPHA, markersize=3)
  mypl.plot(iters2, list2, 'b.-', linewidth=LOSS_LINE_WIDTH - 0.1, alpha=LOSS_ALPHA + 0.1, markersize=2)
  mypl.axis([x_min, x_max, y_min, y_max])

  plt.xticks(fontsize=7)
  plt.yticks(fontsize=7)
  plt.savefig(out_name, dpi=DPI)


def easyDo():
  log_name1 = "1.txt"
  a = resolve_caffe_log(log_name1, data = True, println = True, file = True, pic = True, pic_data = False)
  [train_iters1, train_loss_list1, test_iters1, test_loss_list1, test_accu_list1] = a

  log_name2 = "2.txt"
  b = resolve_caffe_log(log_name2, data = True, println = True, file = True, pic = True, pic_data = False)
  [train_iters2, train_loss_list2, test_iters2, test_loss_list2, test_accu_list2] = b
  
  compare_logs(train_iters1, train_loss_list1, train_iters2, train_loss_list2, "TrainLoss-Red is Origion.png")
  compare_logs(test_iters1, test_loss_list1, test_iters2, test_loss_list2, "TestLoss-Red is Origion.png")
  compare_logs(test_iters1, test_accu_list1, test_iters2, test_accu_list2, "TestAccu-Red is Origion.png")



