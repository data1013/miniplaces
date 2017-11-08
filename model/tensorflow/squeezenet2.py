import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001 #Squeezenet paper says start at 0.04 then decrease linearly, 0.001 is default
dropout = 0.5 # Dropout, probability to keep units
training_iters = 5000 #initially 50,000
step_display = 50 #initially 50
step_save = 1000 #initially 10,000
start_from = ''
last_session = ''
new_session = 'sqz2.ckpt'

f = open("./outputs/datalieSqueeze.txt", "w+")
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

#With reference to implementation here: https://github.com/Khushmeet/squeezeNet
def fire(input, fire_id, channel, s1, e1, e3,):
    weights = {
        'wfr1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
        'wfr2': tf.Variable(tf.truncated_normal([3, 3, s1, e3])),
        'wfr3': tf.Variable(tf.truncated_normal([1, 1, s1, e1]))
    }
        
    biases = {
        'bfr1': tf.Variable(tf.truncated_normal([s1])),
        'bfr2': tf.Variable(tf.truncated_normal([e3])),
        'bfr3': tf.Variable(tf.truncated_normal([e1]))
    }

    with tf.name_scope(fire_id):
        conv1 = tf.nn.conv2d(input, weights['wfr1'], strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bfr1']))
        print 'fire conv1: ', relu1

        conv2 = tf.nn.conv2d(relu1, weights['wfr2'], strides=[1, 1, 1, 1], padding='SAME')
        bias2 = tf.nn.bias_add(conv2, biases['bfr2'])
        print 'fire conv2: ', bias2

        conv3 = tf.nn.conv2d(relu1, weights['wfr3'], strides=[1, 1, 1, 1], padding='SAME')
        bias3 = tf.nn.bias_add(conv3, biases['bfr3'])
        print 'fire conv3: ', bias3

        concat = tf.concat([bias2, bias3], 3)
        print 'fire concat: ', concat
        return tf.nn.relu(concat)
    
def squeezenet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([7, 7, 3, 96])),
        'wc10': tf.Variable(tf.truncated_normal([1, 1, 512, 256])),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }
    '''
    Changed wc1(3) to the 3rd index from the wc1 from alexnet, 100 classes
    '''

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([96])),
        'bc10': tf.Variable(tf.truncated_normal([256])),

        'bo': tf.Variable(tf.ones(100))
    }

    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
    bias1 = tf.nn.bias_add(conv1, biases['bc1'])

    print 'conv1: ', bias1

    maxpool1 = tf.nn.max_pool(bias1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    print 'maxpool1: ', maxpool1

    fire2 = fire(maxpool1, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')

    print 'fire2: ', fire2

    fire3 = fire(fire2, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')

    print 'fire3: ', fire3

    fire4 = fire(fire3, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')

    print 'fire4: ', fire4

    maxpool4 = tf.nn.max_pool(fire4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    print 'maxpool4: ', maxpool4

    fire5 = fire(maxpool4, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')

    print 'fire5: ', fire5

    fire6 = fire(fire5, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')

    print 'fire6: ', fire6

    fire7 = fire(fire6, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')

    print 'fire7: ', fire7

    fire8 = fire(fire7, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')

    print 'fire8: ', fire8

    maxpool8 = tf.nn.max_pool(fire8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    print 'maxpool8: ', maxpool8

    fire9 = fire(maxpool8, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')

    print 'fire9: ', fire9

    dropout9 = tf.nn.dropout(fire9, keep_dropout)

    print 'dropout9: ', dropout9

    conv10 = tf.nn.conv2d(dropout9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')

    print 'conv10: ', conv10

    bias10 = tf.nn.bias_add(conv10, biases['bc10'])

    print 'bias10: ', bias10

    out1 = tf.nn.avg_pool(bias10, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME')

    print 'out1: ', out1

    # Don't actualy want the FC below since SqueezeNet doesn't have fully-connected layers...
    # FC + ReLU + Dropout
    fc6 = tf.reshape(out1, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    print 'fc6: ', fc6
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    print 'fc7: ', fc7

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    print 'out: ', out

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

print "logits: ", logits

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
    # saver.restore(sess, last_session)
    #print("Model restored.")
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        # Run optimization op (backprop)
        #print "images_batch: ", len(images_batch)
        #print "labels_batch: ", labels_batch.size

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

    readFile = open("./outputs/datalieSqueeze.txt")
    lines = readFile.readlines()
    readFile.close()
    w = open("./outputs/datalieSqueeze.txt","w")
    w.writelines([item for item in lines[:-240]])
    w.close()