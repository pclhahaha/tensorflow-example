#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data

data_dir = 'temp'
mnist = input_data.read_data_sets(data_dir)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

train_labels = mnist.train.labels
test_labels = mnist.test.labels

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape = (batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape = eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape = (evaluation_size))

#卷积层
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev= 0.1,
                                               dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features],  dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1,
                                               dtype = tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

#全连接层
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height*conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1,
                                               dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1,
                                               dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev = 0.1, dtype = tf.float32))

def conv_net(input_data):
    #first conv-relu-maxpool layer
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    maxpool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                              strides = [1, max_pool_size1, max_pool_size1, 1], padding = 'SAME')
    #second conv-relu-maxpool layer
    conv2 = tf.nn.conv2d(maxpool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    maxpool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                             strides = [1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    #transform output into a 1*N layer for next fully connected layer
    final_conv_shape = maxpool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(maxpool2, [final_conv_shape[0], final_shape])

    #first fully connected layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    #second fully connected layer
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return (final_model_output)

#create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return (100. * num_correct/batch_predictions.shape[0])

def run():
    model_output = conv_net(x_input)
    test_model_output = conv_net(eval_input)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model_output, labels = y_target))

    prediction = tf.nn.softmax(model_output)
    test_prediction = tf.nn.softmax(test_model_output)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        train_acc = []
        test_acc = []

        for i in range(generations):
            rand_index = np.random.choice(len(train_xdata), size = batch_size)
            rand_x = train_xdata[rand_index]
            rand_x = np.expand_dims(rand_x, 3)
            rand_y = train_labels[rand_index]
            train_dict = {x_input : rand_x, y_target : rand_y}
            sess.run(train_step, train_dict)
            temp_train_loss, temp_train_preds = sess.run([loss, prediction], train_dict)
            temp_train_acc = get_accuracy(temp_train_preds, rand_y)
            if (i+1) % eval_every == 0:
                eval_index = np.random.choice(len(test_xdata), evaluation_size)
                eval_x = test_xdata[eval_index]
                eval_x = np.expand_dims(eval_x, 3)
                eval_y = test_labels[eval_index]
                test_dict = {eval_input : eval_x, eval_target : eval_y}
                test_preds = sess.run(test_prediction, test_dict)
                temp_test_acc = get_accuracy(test_preds, eval_y)

                train_loss.append(temp_train_loss)
                train_acc.append(temp_train_acc)
                test_acc.append(temp_test_acc)
                acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
                acc_and_loss = [np.round(x,2) for x in acc_and_loss]
                print('generation # {}. train loss: {:.2f}. train acc (test acc): {:.2f} ({:.2f})'.
                  format(*acc_and_loss))

if __name__ == '__main__':
    run()










