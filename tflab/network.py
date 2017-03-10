'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from cost_functions import mse, mmd_squared, gaussian, multiscale_gaussian


class FeedForward(object):
    def _parse_list_arg(self, arg, size):
        if isinstance(arg, list):
            if len(arg) != size:
                raise ValueError("Argument list has length {}, need {} ".format(len(arg), size))
            return arg
        else:
            return [arg] * size

    def __init__(self, sizes, nonlinearities=tf.nn.relu, seed=1234):
        self.sizes = sizes
        self._init_weights()
        self.nonlinearities = self._parse_list_arg(nonlinearities, len(sizes) - 1)
        self.rng = np.random
        self.rng.seed(seed)

    def _init_weights(self):
        self.weights = []
        for i in range(len(self.sizes) - 1):
            self.weights.append(tf.Variable(tf.random_normal((self.sizes[i], self.sizes[i + 1])), dtype=tf.float32))

    def transform_(self, X_):
        h = X_
        for i, W in enumerate(self.weights):
            h_raw = tf.matmul(h, W)
            h = self.nonlinearities[i](h_raw)
        return h

    def transform(self, session, X):
        X_ = tf.placeholder(tf.float32, (None, self.sizes[0]))
        Y_hat_ = self.transform_(X_)
        return session.run(Y_hat_, feed_dict={X_: X})

    def train_(self, loss, optimizer):
        return optimizer.minimize(loss)

    def _gen_paired_minibatch(self, batch_size, i, size_limit):
        indicies = range(i * batch_size, (i + 1) * batch_size)
        indicies = [i % size_limit for i in indicies]
        return indicies

    def _gen_unpaired_minibatch(self, batch_size, size_limit):
        indicies = np.random.choice(range(size_limit), [batch_size])
        return indicies


class FeedForwardRegression(FeedForward):
    def loss_(self, X_, Y_):
        Y_hat_ = self.transform_(X_)
        return mse(Y_hat_, Y_)

    def train(self, sess, X, Y,
              steps=10000, minibatch_size=200,
              optimizer=tf.train.RMSPropOptimizer(learning_rate=.001)):
        X_ = tf.placeholder(tf.float32, (None, self.sizes[0]))
        Y_ = tf.placeholder(tf.float32, (None, self.sizes[-1]))
        loss_ = self.loss_(X_, Y_)
        train_ = self.train_(loss_, optimizer)

        sess.run(tf.global_variables_initializer())
        # Fit all training data
        losses = []
        for step in range(steps):
            n_batch = X.shape[0] // minibatch_size + (X.shape[0] % minibatch_size != 0)
            i_batch = (step % n_batch) * minibatch_size
            batch_X = X[i_batch:i_batch + minibatch_size]
            batch_Y = Y[i_batch:i_batch + minibatch_size]
            train_val, loss_val = sess.run([train_, loss_], feed_dict={X_: batch_X, Y_: batch_Y})
            losses.append(loss_val)
            if step % 500 == 0:
                print("Step {} of {}, mse {}".format(step, steps, loss_val))
        return losses


class MMDNet(FeedForward):
    def train(self, sess, X, Y, R, S,
              scale=1.0,
              scale_width=4.0,
              n_scales=5,
              alpha_pair=.1,
              pre_train_steps=500,
              pre_train_optimizer=None,
              steps=10000,
              paired_minibatch_size=20,
              unpaired_minibatch_size=200,

              optimizer=tf.train.RMSPropOptimizer(learning_rate=.001)):
        X_ = tf.placeholder(tf.float32, (None, self.sizes[0]))
        Y_ = tf.placeholder(tf.float32, (None, self.sizes[-1]))
        R_ = tf.placeholder(tf.float32, (None, self.sizes[0]))
        S_ = tf.placeholder(tf.float32, (None, self.sizes[-1]))
        Y_hat_ = self.transform_(X_)
        S_hat_ = self.transform_(R_)

        mse_ = mse(Y_hat_, Y_)
        if pre_train_optimizer is None:
            pre_train_optimizer = optimizer
        train_mse_ = self.train_(mse_, pre_train_optimizer)

        if n_scales > 1:
            scales = list(scale * (10 ** np.linspace(-scale_width / 2, scale_width / 2, num=n_scales)))
            weigths = [1] * len(scales)
            kernel = lambda x, y: multiscale_gaussian(x, y, scales, weigths)
        else:
            kernel = lambda x, y: gaussian(x, y, scale)
        mmd_ = mmd_squared(S_hat_, S_, kernel)
        loss_ = alpha_pair * mse_ + (1 - alpha_pair) * mmd_

        train_ = self.train_(loss_, optimizer)

        sess.run(tf.global_variables_initializer())
        # Fit all training data
        losses = []
        for step in range(pre_train_steps):
            XY_indicies = self._gen_paired_minibatch(paired_minibatch_size, step, X.shape[0])
            batch_X = X[XY_indicies]
            batch_Y = Y[XY_indicies]

            _, loss_val = sess.run([train_mse_, mse_], feed_dict={X_: batch_X, Y_: batch_Y})
            if step % 100 == 0:
                mse_val = sess.run(mse_, feed_dict={X_: batch_X, Y_: batch_Y})
                print("Pre-Training Step {} of {}, loss {}, mse {}".format(step, pre_train_steps, loss_val, mse_val))

        for step in range(steps):
            XY_indicies = self._gen_paired_minibatch(paired_minibatch_size, step, X.shape[0])
            R_indicies = self._gen_unpaired_minibatch(unpaired_minibatch_size, R.shape[0])
            S_indicies = self._gen_unpaired_minibatch(unpaired_minibatch_size, S.shape[0])

            batch_X = X[XY_indicies]
            batch_Y = Y[XY_indicies]
            batch_R = R[R_indicies]
            batch_S = S[S_indicies]
            _, loss_val = sess.run([train_, loss_], feed_dict={X_: batch_X,
                                                               Y_: batch_Y, R_: batch_R, S_: batch_S})
            losses.append(loss_val)
            if step % 500 == 0 or step in {0,1,2,5,10}:
                mmd_val, mse_val = sess.run([mmd_, mse_],
                                            feed_dict={X_: batch_X, Y_: batch_Y, R_: batch_R, S_: batch_S})
                print("Training Step {} of {}, loss {}, mmd {}, mse {}".format(step, steps, loss_val, mmd_val, mse_val))
        return losses
