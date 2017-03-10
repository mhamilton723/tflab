import tensorflow as tf
import os
w2v = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
skipgram_word2vec = w2v.skipgram_word2vec
