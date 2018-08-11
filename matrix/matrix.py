#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
sess = tf.Session()

def test_matrix():
    identity_matrix = tf.diag([1.0, 1.0, 1.0])
    #创建正态分布的矩阵
    A = tf.truncated_normal([2, 3])
    #创建填充的张量或矩阵
    B = tf.fill([2, 3], 5.0)
    C = tf.random_uniform([3, 2])
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))

    print(sess.run(B))
    print(sess.run(identity_matrix))
    #矩阵乘法
    print(sess.run(tf.matmul(B, identity_matrix)))
    #矩阵转置
    print(sess.run(tf.transpose(C)))
    #矩阵行列式
    print(sess.run(tf.matrix_determinant(D)))
    #矩阵逆矩阵
    print(sess.run(tf.matrix_inverse(D)))

if __name__ == '__main__':
    test_matrix()