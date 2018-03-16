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
    from network import FeedForwardSMRegression,FeedForwardRegression
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
    train_y =mnist.target
    
    n_samples = mnist.data.shape[0]
    n_features= mnist.data.shape[1]
    n_classes= len(np.unique(mnist.target))

    train_y=train_y.astype(np.int16)
    train_y=np.eye(n_classes)[train_y]
    return train_x,train_y,n_samples,n_features,n_classes
    

# Parameters
steps = 1000
learning_rate = 0.001
train_x,train_y,n_samples,n_features,n_classes=PrepareData()






rng = np.random
rng.seed(1234)

# Training Data

#y = tf.placeholder(tf.float32,shape=[None,n_classes])
#x = tf.placeholder(tf.float32,shape=[None,n_features])
#
#W = tf.Variable(tf.zeros([n_features,n_classes]))
#b = tf.Variable(tf.zeros([n_classes]))


# declaring model a.k.a. op


#pred = tf.nn.softmax(tf.matmul(x,W,name="matmulp")+b)

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
opt_names = [
        'SGD', 
        'SGD+AS', 
        'RMSProp', 
        'RMSProp+AS', 
        'ADAM', 
        'SGD+M', 
        'SGD+NM', 
        'Adagrad'
        ]

# Launch the graph
losses = []
with tf.Session() as sess:
    for i, opt in enumerate(opts):
        print(opt_names[i])
        reg = FeedForwardSMRegression([784, 10], nonlinearities=lambda x: tf.exp(x)/tf.reduce_sum(tf.exp(x)))
        loss = reg.train(sess, train_x, train_y, minibatch_size=100,
                         steps=steps, optimizer=opts[i])
        losses.append(loss)

plt.clf()
for loss, opt_name in zip(losses, opt_names):
    plt.plot(loss[::5], '+-', alpha=.5, label=opt_name)
plt.legend()
plt.savefig("D:\\BecomingADS\\tflab\\Plots\\new_lr_comparison.png")
