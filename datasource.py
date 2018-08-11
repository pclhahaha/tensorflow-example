#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)

    print(len(mnist.train.images))

    print(len(mnist.test.images))

    print(len(mnist.validation.images))

    print(mnist.train.labels[1,:])

if __name__ == "__main__":
    load_mnist()