

# sudo apt-get install python-imaging-tk
from PIL import ImageTk
# from Tkinter import *
import Tkinter

import os, sys, commands
import cv2
import numpy as np


class IcePicker():

    def __init__(self, folder_path):
        if not os.path.exists(folder_path):
            print 'ERROR: folder not exists! --- ' + folder_path
            sys.exit(-1)
        self.root_path = os.path.abspath(folder_path)

        jpg_list = commands.getoutput('ls ' + os.path.join(self.root_path, '*-pick.jpg')).split()
        pick_list = commands.getoutput('ls ' + os.path.join(self.root_path, '*.pick')).split()
        if len(jpg_list) != len(pick_list):
            print 'ERROR: *-pick.jpg *.pick not equal! --- ' + folder_path
            sys.exit(-1)
        
        self.abs_names = [i[0:-9] for i in jpg_list]
        self.boxes = []  # self.boxes[i] is ndarray of (pickN, 4)
        for ii in self.abs_names:
            fPick = open(ii + '.pick', 'r')
            pickList = [i.strip().split() for i in fPick.readlines()]
            fPick.close()
            self.boxes.append(self._strMatrix2intMatrix(pickList))

        self.checkID = -1
        self.oldWidth = -1
        self.oldHeight = -1
        self.sideLen = 100
        self.ice = []  # self.ice[ice_index] = [x1, y1, x2, y2]. This contains only for single img.

        self.exit = False
        self.scale = 4
        self.canvas = None

    def _strMatrix2intMatrix(self, strMatrix):
        results = np.zeros((len(strMatrix), 4), dtype=np.int32)
        for i in range(0, len(strMatrix)):
            ii = strMatrix[i]
            results[i] = [int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3])]
        return results

    def pick(self):
        for ii in range(0, len(self.abs_names)):
            if self.exit:
                break
            if not os.path.exists(self.abs_names[ii] + '-small-' + str(self.scale) + '.jpg'):
                self.checkID = ii
                self.PickForSingleImg(self.abs_names[ii])

    def PickForSingleImg(self, img_path):
        img0 = cv2.imread(img_path + '-pick.jpg')
        width0 = img0.shape[0]
        height0 = img0.shape[1]
        self.oldWidth = int(width0)
        self.oldHeight = int(height0)
        self.ice = []
        img1 = cv2.resize(img0, (int(width0 / self.scale), int(height0 / self.scale)))
        cv2.imwrite(img_path + '-small-' + str(self.scale) + '.jpg', img1)

        root = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(root, width = 1000, height = 1000)
        imgtk = ImageTk.PhotoImage(file = img_path + '-small-' + str(self.scale) + '.jpg')
        self.canvas.create_image(500, 500, image = imgtk)
        self.canvas.bind('<Button-1>', self._clickEvent)
        self.canvas.bind('<Key>', self._keyEvent)
        self.canvas.pack()
        Tkinter.mainloop()

        fIce = open(img_path + '.ice', 'w+')
        for i_ice in range(0, len(self.ice)):
            fIce.writelines(str(self.ice[i_ice][0]) + ' ' + str(self.ice[i_ice][1]) + ' ' + str(self.ice[i_ice][2]) + ' ' + str(self.ice[i_ice][3]) + '\r\n')
        fIce.close()
        self.ice = []

    def _clickEvent(self, e):
        print ' '
        print e.x * self.scale
        print e.y * self.scale
        X = float(e.x)
        Y = float(e.y)
        for i_pick in range(0, self.boxes[self.checkID].shape[0]):
            x1 = self.boxes[self.checkID][i_pick, 0]
            y1 = self.boxes[self.checkID][i_pick, 1]
            x2 = self.boxes[self.checkID][i_pick, 2]
            y2 = self.boxes[self.checkID][i_pick, 3]
            # There is one box containing the click-point
            if x1 <= (self.scale * X) <= x2 and y1 <= (self.scale * Y) <= y2:
                print 'Found!\tImgID:' + str(self.checkID) + '\tPickID:' + str(i_pick)
                print 'X=[' + str(x1) + '\t' + str(x2) + ']. Y=[' + str(y1) + '\t' + str(y2) + '].'
                self.ice.append([x1, y1, x2, y2])
                self.canvas.create_rectangle(x1/self.scale, y1/self.scale, x2/self.scale, y2/self.scale, width = 3, outline = 'green')
                return

        # There is no box containing the click-point.
        x1 = max(0, min(self.oldWidth - self.sideLen, int(X * self.scale - self.sideLen / 2)))
        y1 = max(0, min(self.oldHeight - self.sideLen, int(Y * self.scale - self.sideLen / 2)))
        x2 = x1 + self.sideLen
        y2 = y1 + self.sideLen
        print 'New point!\tImgID' + str(self.checkID)
        print 'X=[' + str(x1) + '\t' + str(x2) + ']. Y=[' + str(y1) + '\t' + str(y2) + '].'
        self.ice.append([x1, y1, x2, y2])
        self.canvas.create_rectangle(x1/self.scale, y1/self.scale, x2/self.scale, y2/self.scale, width = 3, outline = 'green')

    def _keyEvent(self, e):
        print e.char
        self.exit = True


