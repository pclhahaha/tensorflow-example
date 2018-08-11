#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def regress():
    sess = tf.Session()
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1]))

    my_output = tf.multiply(x_data, A)
    loss = tf.square(my_output - y_target)

    init = tf.initialize_all_variables()
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    train_step = my_opt.minimize(loss)