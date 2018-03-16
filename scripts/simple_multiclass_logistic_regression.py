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

from network import FeedForwardRegression
from  optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer


#Setting parameters
learning_late= 0.001
num_of_epochs =50
batch_size=100
display_step=1

#declaring data placeholders

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


y = tf.placeholder(tf.float32,shape=[None,10])
x = tf.placeholder(tf.float32,shape=[None,784])

# declaring variables

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# declaring model a.k.a. op

pred = tf.nn.softmax(tf.matmul(x,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-8),reduction_indices=1))
#optimizer = tf.train.GradientDescentOptimizer(learning_late).minimize(cost)
optimizer = ASGradientDescentOptimizer(learning_late,scale=1.0001).minimize(cost)
#optimizer = ASRMSPropOptimizer(learning_late).minimize(cost)
# declaring initializing variables op

init=tf.global_variables_initializer()

losses=[]
ASG_Losses=[]

with tf.Session() as sess :
    sess.run(init)
    
    for epoch in range(num_of_epochs):
        avgcost=0
        num_of_batches= int(mnist.train.num_examples/batch_size)
        
        for batchnum in range(num_of_batches):
            x_minibatch,y_minibatch= mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:x_minibatch ,y:y_minibatch})
            avgcost+= c/batch_size
            
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avgcost))
        ASG_Losses.append(avgcost)
    print("Optimization Done")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(   {x: mnist.test.images, y: mnist.test.labels}))

losses.append(ASG_Losses)
SGD_Losses=[]

y1 = tf.placeholder(tf.float32,shape=[None,10])
x1 = tf.placeholder(tf.float32,shape=[None,784])

# declaring variables

W1 = tf.Variable(tf.zeros([784,10]))
b1 = tf.Variable(tf.zeros([10]))
pred1 = tf.nn.softmax(tf.matmul(x1,W1)+b1)
cost1 = tf.reduce_mean(-tf.reduce_sum(y1*tf.log(pred1+1e-8),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_late).minimize(cost1)



init1=tf.global_variables_initializer()

with tf.Session() as sess1 :
    sess1.run(init1)
    
    for epoch in range(num_of_epochs):
        avgcost=0
        num_of_batches= int(mnist.train.num_examples/batch_size)
        for batchnum in range(num_of_batches):
            x_minibatch,y_minibatch= mnist.train.next_batch(batch_size)
            _,c=sess1.run([optimizer,cost1],feed_dict={x1:x_minibatch ,y1:y_minibatch})
            avgcost+= c/batch_size
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avgcost))
        SGD_Losses.append(avgcost)
    print("Optimization Done")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(   {x: mnist.test.images, y: mnist.test.labels}))

losses.append(SGD_Losses)




plt.clf()
for loss, opt_name in zip(losses, ['ASGD','SGD']):
    plt.plot(loss[::5], '+-', alpha=.5, label=opt_name)
plt.legend()
#plt.savefig("../../tflab/plots/logreg_comparison1.png")

plt.savefig("D:/BecomingADS/tflab/plots/logreg_comparisonzz.png")




