# append to ~/.bashrc: export PYTHONSTARTUP=~/.pythonstartup

import readline, rlcompleter
readline.parse_and_bind("tab: complete")  # for Linux
# readline.parse_and_bind("bind ^I rl_complete")  # for Mac

import os
cl='os.system("clear")'
# exec(cl)

# import sys
# sys.path.insert(0, "/home/xyf/caffe-master/python")


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


