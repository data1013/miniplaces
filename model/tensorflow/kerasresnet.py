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

# Dataset Parameters
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

fwrite1 = open("./outputs/trainingloss.txt", "w")

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'training': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'training': False
    }

opt_data_test = {
    'data_root': '../../data/images/',
    'data_list': '../../data/test.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'training': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_test)

model = ResNet50(
    include_top = True,
    weights = None,
    input_shape = (224, 224, 3),
    pooling = None,
    classes = 100
)

#default top_k is k = 5
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy', 'top_k_categorical_accuracy'])

training_iters = 1
batch_size = 16 #need to be small to not run out of memory, like 32

step = 0

while step < training_iters:
    step = step + 1
    #training in batches
    images_batch, labels_batch = loader_train.next_batch(batch_size)
    labels_batch = to_categorical(labels_batch, 100)
    train_loss = model.train_on_batch(images_batch, labels_batch)

    print('train loss = ' + str(train_loss[0]))

    if step % 50 == 0:
        fwrite1.write(str(train_loss[0])+"\n")
        print('Saved training loss iter = ' + str(step))
    if step % 500 == 0:
        print('saved model')
        model.save('./resnetModels/resnet50_%09d' % str(step))

fwrite1.close()