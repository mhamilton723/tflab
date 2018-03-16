import numpy as np
import tensorflow as tf

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from os import path
#sys.path.append('D:\\ML-Lab\\tflab\\tflab')
libpath="../tflab/tflab"
if (libpath not in sys.path):
    sys.path.append("../tflab/tflab")

try:
    
    #from tflab.network import FeedForwardRegression
    #from tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer
    #import "D:/ML-Lab/tflab/tflab/tflab/"
    #from tflab.network import FeedForwardRegression
    #from tflab.network import ASGradientDescentOptimizer, ASRMSPropOptimizer
    from network import FeedForwardRegression
    from  optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer

    #from tflab.network import FeedForwardRegression
    #from tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer
except ImportError:
    print("Error in importing...")
    from tflab.tflab.network import FeedForwardRegression
    from tflab.tflab.optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer

# Parameters
steps = 10000
learning_rate = 0.001
n_samples = 2000
X_dim = 200
Y_dim = 100

rng = np.random
rng.seed(1234)
# Training Data
train_X = rng.randn(n_samples, X_dim)
M = rng.randn(X_dim, Y_dim)

train_Y = np.dot(train_X, M)

opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate),
    tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    ASRMSPropOptimizer(base_learning_rate=learning_rate),
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
        reg = FeedForwardRegression([X_dim, Y_dim], nonlinearities=lambda x: x)
        loss = reg.train(sess, train_X, train_Y, minibatch_size=20,
                         steps=steps, optimizer=opts[i])
        losses.append(loss)

plt.clf()
for loss, opt_name in zip(losses, opt_names):
    plt.plot(loss[::100], '+-', alpha=.5, label=opt_name)
plt.legend()
plt.savefig("D:\\BecomingADS\\tflab\\Plots\\lr_comparison.png")
