import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from tensorflow.python.saved_model import builder as saved_model_builder

# Dataset Parameters
batch_size = 256
load_size = 128
fine_size = 112
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 10000 #initially 50,000
step_display = 50 #initially 50
step_save = 2500 #initially 10,000
export_dir = 'builtModel/'
start_from = ''
last_session = '8000.ckpt-8000'
new_session = '.ckpt'

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
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([7, 7, 3, 72], stddev=np.sqrt(2./(7*7*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 72, 192], stddev=np.sqrt(2./(5*5*72)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 192, 336], stddev=np.sqrt(2./(3*3*192)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 336, 192], stddev=np.sqrt(2./(3*3*336)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 192, 192], stddev=np.sqrt(2./(3*3*192)))),

        'wf6': tf.Variable(tf.random_normal([9408, 3840], stddev=np.sqrt(2./(9408)))),
        'wf7': tf.Variable(tf.random_normal([3840, 3840], stddev=np.sqrt(2./3840))),
        'wo': tf.Variable(tf.random_normal([3840, 100], stddev=np.sqrt(2./3840)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out

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
    'training': True
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
logits = alexnet(x, keep_dropout, train_phase)

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
    #saver.restore(sess, last_session)
    #print("Model restored.")
    
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

            if acc5 > 0.80:
                current_step = str(step)

                f = open("./outputs/datalieALEXFINAL-"+current_step+".txt", "w+")
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

                readFile = open("./outputs/datalieALEXFINAL-"+current_step+".txt")
                lines = readFile.readlines()
                readFile.close()
                w = open("./outputs/datalieALEXFINAL-"+current_step+".txt","w")
                w.writelines([item for item in lines[:-240]])
                w.close()

        # Save model
        if step % step_save == 0:
            saver.save(sess, str(step)+new_session, global_step=step)
            print "Model saved at Iter %d !" %(step)
        
    print "Optimization Finished!"

    fwrite1.close()
    fwrite2.close()
    fwrite3.close()
    fwrite4.close()
    fwrite5.close()
    fwrite6.close()

    # Evaluate on the whole test set

    current_step = str(step)

    f = open("./outputs/datalieALEXFINAL-"+current_step+".txt", "w+")
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

    readFile = open("./outputs/datalieALEXFINAL-"+current_step+".txt")
    lines = readFile.readlines()
    readFile.close()
    w = open("./outputs/datalieALEXFINAL-"+current_step+".txt","w")
    w.writelines([item for item in lines[:-240]])
    w.close()
