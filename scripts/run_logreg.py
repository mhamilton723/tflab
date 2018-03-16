# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:24:41 2018

@author: psundara
"""

import numpy as np
import tensorflow as tf

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from os import path

libpath="../tflab/tflab"
if (libpath not in sys.path):
    sys.path.append("../tflab/tflab")

try:
    
    #from tflab.network import FeedForwardRegression
    #from tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer
    #import "D:/ML-Lab/tflab/tflab/tflab/"
    #from tflab.network import FeedForwardRegression
    #from tflab.network import ASGradientDescentOptimizer, ASRMSPropOptimizer
    from network import FeedForwardSMRegression
    from  optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer

    #from tflab.network import FeedForwardRegression
    #from tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer
except ImportError:
    print("Error in importing...")
    from tflab.tflab.network import FeedForwardRegression
    from tflab.tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer


def PrepareData():
    import sklearn
    from sklearn.datasets import fetch_mldata
    custom_data_home ="D/BecomeADS/tflab/data/"
    mnist=fetch_mldata('MNIST original', data_home=custom_data_home)
    train_x = mnist.data
    train_x=train_x/255
    train_y =mnist.target.reshape(mnist.target.shape[0],1)
    n_samples = mnist.data.shape[0]
    n_features= mnist.data.shape[1]
    return train_x,train_y,n_samples,n_features
    

# Parameters
steps = 10000
learning_rate = 0.001
train_x,train_y,n_samples,n_features=PrepareData()
X_dim = 200
Y_dim = 100

rng = np.random
rng.seed(1234)

# Training Data

y = tf.placeholder(tf.float32,shape=[None,10])
x = tf.placeholder(tf.float32,shape=[None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# declaring model a.k.a. op

pred = tf.nn.softmax(tf.matmul(x,W)+b)

#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-8),reduction_indices=1))

opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate,scale=1.001),
    tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    ASRMSPropOptimizer(base_learning_rate=learning_rate,scale=1.001),
    tf.train.AdamOptimizer(learning_rate=learning_rate),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
    tf.train.AdagradOptimizer(learning_rate=learning_rate)
]
opt_names = ['SGD', 'SGD+AS', 'RMSProp', 'RMSProp+AS', 'ADAM', 'SGD+M', 'SGD+NM', 'Adagrad']

# Launch the graph
losses = []
with tf.Session() as sess:
    for i, opt in enumerate(opts):
        print(opt_names[i])
        reg = FeedForwardSMRegression([784, 10], nonlinearities=lambda x: x)
        loss = reg.train(sess, train_x, train_y, minibatch_size=100,
                         steps=steps, optimizer=opts[i])
        losses.append(loss)

plt.clf()
for loss, opt_name in zip(losses, opt_names):
    plt.plot(loss[::100], '+-', alpha=.5, label=opt_name)
plt.legend()
plt.savefig("D:\\BecomingADS\\tflab\\Plots\\new_lr_comparison.png")
