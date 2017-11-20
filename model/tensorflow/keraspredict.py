from __future__ import division

import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from tensorflow.python.saved_model import builder as saved_model_builder
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical

from scipy.misc import imread, imresize

import glob
import os

list_of_files = glob.glob("./resnetModels/*") # * means all if need specific format then *.csv
latest_file = max(list_of_files)

model = load_model(latest_file)
paths = os.listdir("../../data/images/test")

data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

load_size = 256
fine_size = 224
c = 3

f = open("./outputs/datalieRESNET.txt", "w")

# Process each image in the test/ directory separately (no batching)
for i, path in enumerate(paths):
    image = imread("../../data/images/test/" + path)
    # ResNet50 takes load_size x load_size x 3
    image = imresize(image, (load_size, load_size))

    #DataLoader.py process
    image = image.astype(np.float32)/255.
    image = image - data_mean
    offset_h = (load_size-fine_size)//2
    offset_w = (load_size-fine_size)//2

    image = image[offset_h:offset_h+fine_size, offset_w:offset_w+fine_size, :]
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    #get the prediction likelihood values from each class for the given image
    predictions = model.predict(image)
    predsort = np.argsort(predictions)[0]
    top5 = list(predsort[95:100])[::-1]

    print("Image: " + str(path) + " " + str(top5[0]) + " " + str(top5[1]) + " " + str(top5[2]) + " " + str(top5[3]) + " " + str(top5[4]))

    f.write("test/" + str(path) + " %i %i %i %i %i\n" % (top5[0], top5[1], top5[2], top5[3], top5[4]))

    #print("Image: " + path + ".jpg %i %i %i %i %i\n" % (top5[0], top5[1], top5[2], top5[3], top5[4]))

f.close()