import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

f= open("datalie.txt","w+")

a = [[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]
imgCounter = 1

for x in xrange(0, len(a)):
    imgFile = ""
    if imgCounter < 10:
        imgFile = "0000000"+str(imgCounter)
    elif imgCounter < 100:
        imgFile = "000000"+str(imgCounter)
    elif imgCounter < 1000:
        imgFile = "00000"+str(imgCounter)
    elif imgCounter < 10000:
        imgFile = "0000"+str(imgCounter)
    elif imgCounter < 100000:
        imgFile = "000"+str(imgCounter)
    #print("test/"+imgFile+".jpg")
    imgCounter = imgCounter + 1
    f.write(str(a[x][0])+" "+str(a[x][1])+" "+str(a[x][2])+" "+str(a[x][3])+" "+str(a[x][4])+"\n")

f.close() 