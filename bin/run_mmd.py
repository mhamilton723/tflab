import matplotlib as mpl
import numpy as np
import tensorflow as tf

mpl.use('Agg')
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tflab.network import FeedForwardRegression, MMDNet


def gen_synthetic_data(n_samples, X_dim, Y_dim):
    X = np.random.normal(0., 1., size=[n_samples, X_dim])
    W = np.random.normal(0., 1., size=[X_dim, Y_dim])
    Y = np.dot(X, W) + np.random.normal(0, .1, size=[n_samples, Y_dim])
    return X, Y, W


# Parameters
alpha_pair = .0001
angle_of_rotation = 255
scale = 1.
steps = 12000
pre_train_steps = 10000
learning_rate = 0.001
n_samples = 100000
n_labeled = 50
paired_minibatch_size = min(100, n_labeled)
unpaired_minibatch_size= 200
X_dim = 30
Y_dim = 30
plot = False
np.random.seed(12345)

X, Y, W = gen_synthetic_data(n_samples, X_dim, Y_dim)
train_X = X[0:n_labeled]
train_Y = Y[0:n_labeled]
R = X[n_labeled:]
S = Y[n_labeled:]

# Launch the graph
losses = []
with tf.Session() as sess:
    reg_mmd = MMDNet([X_dim, Y_dim], nonlinearities=lambda x: x)
    loss = reg_mmd.train(sess, train_X, train_Y, R, S,
                         scale=scale,
                         alpha_pair=alpha_pair,
                         paired_minibatch_size=paired_minibatch_size,
                         unpaired_minibatch_size=200,
                         pre_train_steps=pre_train_steps,
                         steps=steps - pre_train_steps,
                         pre_train_optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate),
                         optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    losses.append(loss)
    Y_hat_mmd = reg_mmd.transform(sess, X)

    reg = FeedForwardRegression([X_dim, Y_dim], nonlinearities=lambda x: x)
    loss = reg.train(sess, train_X, train_Y,
                     minibatch_size=paired_minibatch_size,
                     steps=steps,
                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    losses.append(loss)
    Y_hat = reg.transform(sess, X)

    reg_all = FeedForwardRegression([X_dim, Y_dim], nonlinearities=lambda x: x)
    loss = reg_all.train(sess, X, Y,
                         minibatch_size=paired_minibatch_size,
                         steps=steps,
                         optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    losses.append(loss)
    Y_hat_all = reg_all.transform(sess, X)


def mse(X, Y):
    return np.sqrt(np.square(X - Y)).mean()


print "LR mse {} ".format(mse(Y_hat, Y))
print "LR ALL mse {} ".format(mse(Y_hat_all, Y))
print "MMD mse {} ".format(mse(Y_hat_mmd, Y))
