from __future__ import print_function
import collections
import inspect
import os
import zipfile

import h5py
import numpy as np
import tensorflow as tf

import tflab.tflab.optimizers


def dot(x, y):
    return tf.reduce_sum(x * y, axis=-1)


def l2_norm(x):
    return tf.reduce_sum(tf.square(x), axis=-1)


def distance_to_line(a, n, p):
    """
    Given a line of form x = a +t*n where t is arbitrary, and given p a point
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    compute the distance from p to the nearest point of x



    :return: distance from p to the nearest point of x
    """
    res = l2_norm((a - p) - tf.expand_dims(dot((a - p), n), -1) * n)
    # nodes = [a - p, dot((a - p), n), tf.expand_dims(dot((a - p), n),-1)]
    # res = tf.Print(res, [tf.shape(n) for n in nodes], "in dist function")
    return res


def get_or_create_path(*paths, **kwargs):
    if "exclude_last" not in kwargs or kwargs["exclude_last"]:
        dir_paths = [path for path in paths][:-1]
    else:
        dir_paths = [path for path in paths]

    if not os.path.exists(os.path.join(*dir_paths)):
        print("making dir {}".format(os.path.join(*dir_paths)))
        os.makedirs(os.path.join(*dir_paths))
    return os.path.join(*paths)


def read_data(filename, n=None, zip=False):
    """Extract the first file enclosed in a zip file as a list of words"""
    if zip:
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    else:
        with open(filename, 'r') as f:
            data = tf.compat.as_str(f.read()).split()
    if n is None:
        return data
    else:
        return data[0:n]


def load_embs(filename, top_n=None):
    with h5py.File(filename, 'r') as hf:
        words = np.array(hf.get('words'))
        embeddings = np.array(hf.get('embeddings'))
        nce_weights = np.array(hf.get('nce_weights'))
        nce_biases = np.array(hf.get('nce_biases'))
        reverse_dictionary = {i: w for (i, w) in enumerate(words)}
    if top_n is not None:
        reverse_dictionary = {i: w for (i, w) in reverse_dictionary.iteritems() if i < top_n}
        embeddings = embeddings[:top_n]
        nce_weights = nce_weights[:top_n]
        nce_biases = nce_biases[:top_n]
    return reverse_dictionary, embeddings, nce_weights, nce_biases


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def cross_reference(dictionary_target, dictionary_source):
    cross_ref = {}
    for (kt, vt) in dictionary_target.iteritems():
        if kt in dictionary_source:
            cross_ref[dictionary_source[kt]] = vt
    return cross_ref


def save_embs(name, reverse_dictionary, embs, nce_weights, nce_biases):
    words = np.array([reverse_dictionary[i] for i in range(len(reverse_dictionary))])
    with h5py.File(name, 'w') as hf:
        hf.create_dataset('words', data=words)
        hf.create_dataset('embeddings', data=embs)
        hf.create_dataset('nce_weights', data=nce_weights)
        hf.create_dataset('nce_biases', data=nce_biases)


def create_true_pairing(cross_ref, nrs):
    p = np.zeros((nrs.vocabulary_size_source, nrs.vocabulary_size_target))
    for (vs, vt) in cross_ref.iteritems():
        p[vs, vt] = 1.0
    return p


def create_sparse_initialization(cross_ref, nrs):
    indicies = []
    values = []
    for (vs, vt) in cross_ref.iteritems():
        indicies.append([vs, vt])
        values.append(1.0)
    return indicies, values, [nrs.vocabulary_size_source, nrs.vocabulary_size_target]


def avg_correct_probability_mass(p_hat, cross_ref):
    probs = np.array([p_hat[vs, vt] for (vs, vt) in cross_ref.iteritems()])
    return probs.mean()


def call_with_flags(function, FLAGS, **kwargs):
    argnames = {arg for arg in inspect.getargspec(function)[0] if arg not in {"self"}}
    params = {k: v for (k, v) in FLAGS.__dict__['__flags'].iteritems() if k in argnames}
    params.update(kwargs)
    for arg in argnames:
        if arg not in params.keys():
            raise UserWarning("argument: {} is not handled by flags, beware of errors".format(arg))
    return function(**params)


def instantiate_with_flags(clazz, FLAGS, **kwargs):
    argnames = {arg for arg in inspect.getargspec(clazz.__init__)[0] if arg not in {"self"}}
    params = {k: v for (k, v) in FLAGS.__dict__['__flags'].iteritems() if k in argnames}
    params.update(kwargs)
    for arg in argnames:
        if arg not in params.keys():
            raise UserWarning("argument: {} is not handled by flags, beware of errors".format(arg))
    return clazz(**params)


def optimize_(loss_, name, starting_lr, decay, global_step_, var_list=None):
    """Build the graph to optimize the loss function."""
    if decay == "exponential":
        lr_ = tf.train.exponential_decay(starting_lr, global_step_, 10000, .98)
    elif decay == "linear":
        lr_ = tf.train.polynomial_decay(starting_lr, global_step_, 1000000, .0001)
    elif decay is None:
        lr_ = tf.convert_to_tensor(starting_lr)
    else:
        raise ValueError("improper decay name {}".format(decay))

    if name == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(lr_)
    elif name == "adam":
        optimizer = tf.train.AdamOptimizer(lr_)
    elif name == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(lr_)
    elif name == "assgd":
        optimizer = optimizers.ASGradientDescentOptimizer(starting_lr)
    else:
        raise ValueError("improper optimizer_name {}".format(name))

    return optimizer.minimize(loss_, global_step=global_step_,
                              gate_gradients=optimizer.GATE_NONE, var_list=var_list), lr_
