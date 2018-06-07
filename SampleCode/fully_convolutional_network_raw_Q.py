""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = 

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
display_step = 10

# Network Parameters
num_input =  # MNIST data input (img shape: 28*28)
num_classes =  # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [])
Y = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weight & bias
# initializer = tf.contrib.layers.xavier_initializer()
weights = {
    
    # initializer of various function
    'wc1': tf.Variable(initializer([])),
    'wc2': tf.Variable(initializer([])),
    'wc3': tf.Variable(initializer([])),
    'wc4': tf.Variable(initializer([])),
    'wc5': tf.Variable(initializer([])),
    
    'out': tf.Variable(initializer([]))
}

biases = {
    'bc1': tf.Variable(initializer([])),
    'bc2': tf.Variable(initializer([])),
    'bc3': tf.Variable(initializer([])),
    'bc4': tf.Variable(initializer([])),
    'bc5': tf.Variable(initializer([])),
    'out': tf.Variable(initializer([]))
}


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
                          
                          
# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ## for fully-convolutional NN
    conv1 = conv2d() #14
    conv2 = conv2d() #7
    conv3 = conv2d() #4
    conv4 = conv2d() #2
    conv5 = conv2d() #1

    
    # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.nn.conv2d() + biases['out'] #1
    
    out = tf.reshape(out, [-1,num_classes])
    
    return out


# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=, labels=))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: , Y: , keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: ,
                                                                 Y: ,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))
