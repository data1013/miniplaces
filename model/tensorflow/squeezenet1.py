import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from tensorflow.python.saved_model import builder as saved_model_builder

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 1 #initially 50,000
step_display = 1 #initially 50
step_save = 1000 #initially 10,000
export_dir = 'builtModel/'
start_from = ''
last_session = '10000.ckpt-1300'
new_session = '11000.ckpt'

f = open("./outputs/datalieALEXFINAL.txt", "w+")
fwrite1 = open("./outputs/trainingloss.txt", "w+")
fwrite2 = open("./outputs/trainingacc1.txt", "w+")
fwrite3 = open("./outputs/trainingacc5.txt", "w+")
fwrite4 = open("./outputs/validationloss.txt", "w+")
fwrite5 = open("./outputs/validationacc1.txt", "w+")
fwrite6 = open("./outputs/validationacc5.txt", "w+")

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def fire_module(input, fire_id, channel, s1, e1, e3,):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e3: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor
    """

    fire_weights = {'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
                    'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1])),
                    'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, s1, e3]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
                   'conv_e_3': tf.Variable(tf.truncated_normal([e3]))}

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def squeezenet(input, classes):
    """
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor
    """

    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 1, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes]))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes]))}

    output = tf.nn.conv2d(input, weights['conv1'], strides=[1,2,2,1], padding='SAME', name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    output = fire_module(output, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')
    output = fire_module(output, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')
    output = fire_module(output, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    output = fire_module(output, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')
    output = fire_module(output, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')
    output = fire_module(output, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    output = fire_module(output, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')

    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout9')

    output = tf.nn.conv2d(output, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
    output = tf.nn.bias_add(output, biases['conv10'])

    output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')

    return output

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/train.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'training': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/val.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'training': False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
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
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = squeezenet(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
# in_top_k Gives a boolean for each image, for whether or not it's correct classification was in the top k outputs
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

accuracy1_nums = tf.nn.in_top_k(logits, y, 1)
accuracy5_nums = tf.nn.in_top_k(logits, y, 5)

# define initialization
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Pre-running init.")
    # Initialize variables
    sess.run(init)

    print("Post-running init.")

    # Restore model weights from previously saved model
    saver.restore(sess, last_session)
    print("Model restored.")
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
        step += 1

        if step % step_display == 0:
            print '[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 

            print "-Iter " + str(step) + ", Training Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(acc1) + ", Top5 = " + \
            "{:.4f}".format(acc5)

            fwrite1.write("{:.6f}".format(l)+"\n")
            fwrite2.write("{:.4f}".format(acc1)+"\n")
            fwrite3.write("{:.4f}".format(acc5)+"\n")

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print "-Iter " + str(step) + ", Validation Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(acc1) + ", Top5 = " + \
            "{:.4f}".format(acc5)

            fwrite4.write("{:.6f}".format(l)+"\n")
            fwrite5.write("{:.4f}".format(acc1)+"\n")
            fwrite6.write("{:.4f}".format(acc5)+"\n")
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, new_session, global_step=step)
            print "Model saved at Iter %d !" %(step)
        
    print "Optimization Finished!"

    fwrite1.close()
    fwrite2.close()
    fwrite3.close()
    fwrite4.close()
    fwrite5.close()
    fwrite6.close()

    # Evaluate on the whole test set
    print 'Evaluation on the whole test set...'
    num_batch = loader_test.size()/batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_test.reset()

    imgCounter = 1
    for i in range(num_batch+1):
        images_batch, labels_batch = loader_test.next_batch(batch_size)    

        feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}

        topprediction = tf.nn.top_k(logits, 5)
        best_vals, best_indices = sess.run(topprediction, feed_dict)

        for img_idx in xrange(len(best_indices)):
            current_image = best_indices[img_idx]

            imgFile = str(imgCounter)
            imgFile = imgFile.zfill(8)
            
            imgCounter += 1

            f.write("test/" + imgFile + ".jpg %i %i %i %i %i\n" % (current_image[0], current_image[1], current_image[2], current_image[3], current_image[4]))

    f.close()

    readFile = open("./outputs/datalieALEXFINAL.txt")
    lines = readFile.readlines()
    readFile.close()
    w = open("./outputs/datalieALEXFINAL.txt","w")
    w.writelines([item for item in lines[:-240]])
    w.close()