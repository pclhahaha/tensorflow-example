#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def multiplication_gate():
    sess = tf.Session()

    a = tf.Variable(tf.constant(4.))
    x_val = 5
    x_data = tf.placeholder(dtype=tf.float32)

    multiply = tf.multiply(a, x_data)
    loss = tf.square(tf.subtract(multiply, 50.))
    init = tf.initialize_all_variables()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)
    print("optimizing a multiplication gate output to 50.")
    for i in range(20):
        sess.run(train_step, feed_dict={x_data:x_val})
        a_val = sess.run(a)
        multi_out = sess.run(multiply, feed_dict={x_data:x_val})
        print(str(a_val) + " * " + str(x_val)+' = '+str(multi_out))

def two_gate():
    sess = tf.Session()

    a = tf.Variable(tf.constant(4.))
    b = tf.Variable(tf.constant(4.))
    x_val = 5
    x_data = tf.placeholder(dtype=tf.float32)

    two_gate = tf.add(tf.multiply(a, x_data), b)
    loss = tf.square(tf.subtract(two_gate, 50.))

    init = tf.initialize_all_variables()
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)
    print("optimizing a two gate output to 50.")
    for i in range(20):
        sess.run(train_step, feed_dict={x_data:x_val})
        a_val = sess.run(a)
        b_val = sess.run(b)
        multi_out = sess.run(two_gate, feed_dict={x_data:x_val})
        print(str(a_val) + " * " + str(x_val)+' + '+str(b_val)+' = '+str(multi_out))

if __name__ == "__main__":
    multiplication_gate()
    two_gate()