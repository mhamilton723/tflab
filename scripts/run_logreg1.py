# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:10:34 2018

@author: psundara
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
import sys
from os import path


#sys.path.append("D:\\BecomingADS\\tflab\\tflab")
libpath="../tflab/tflab"
if (libpath not in sys.path):
    sys.path.append("../tflab/tflab")

#from network import FeedForwardRegression
from  optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer


#Setting parameters
learning_rate= 0.001
num_of_epochs =100
batch_size=100
display_step=20

#declaring data placeholders

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
y = tf.placeholder(tf.float32,shape=[None,10],name="YY")
x = tf.placeholder(tf.float32,shape=[None,784],name="XX")

# declaring variables

W = tf.Variable(tf.zeros([784,10]),name="Weights")
b = tf.Variable(tf.zeros([10]),name="bias")

# declaring model a.k.a. op
pred = tf.nn.softmax(tf.matmul(x,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-8),reduction_indices=1))

opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate,scale=1.0001),
    tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    ASRMSPropOptimizer(base_learning_rate=learning_rate,scale=1.0001),
#    tf.train.AdamOptimizer(learning_rate=learning_rate),
#    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
#    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
#    tf.train.AdagradOptimizer(learning_rate=learning_rate)
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
losses=[]

with tf.Session() as sess:
    
    for i,opt in enumerate(opts):
        templ=[]
        print(opt_names[i])
        optimizer=opt.minimize(cost)
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_of_epochs):
            avgcost=0
            num_of_batches= int(mnist.train.num_examples/batch_size)
            
            for batchnum in range(num_of_batches):
                x_minibatch,y_minibatch= mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,cost],feed_dict={x:x_minibatch ,y:y_minibatch})
                avgcost+= c/batch_size
                
            
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avgcost))
            templ.append(avgcost)
        losses.append(templ)
    



plt.clf()
for loss, opt_name in zip(losses, opt_names):
    plt.plot(loss[::5], '+-', alpha=.5, label=opt_name)
plt.legend()


plt.savefig("D:/BecomingADS/tflab/plots/logreg_comparison_all_27.png")
#plt.savefig("../tflab/plots/logreg_comparison_dynamic.png")



