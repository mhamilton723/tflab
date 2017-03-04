import collections
import inspect
import os
import zipfile

import h5py
import numpy as np
import tensorflow as tf

import optimizers


def get_or_create_path(*paths, **kwargs):
    if "exclude_last" not in kwargs or not kwargs["exclude_last"]:
        dir_paths = [path for path in paths][:-1]
    else:
        dir_paths = [path for path in paths]

    if not os.path.exists(os.path.join(*dir_paths)):
        print "making dir {}".format(os.path.join(*dir_paths))
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


def params_to_name(params):
    sorted_items = sorted(params.iteritems())
    name_list = []
    for k, v in sorted_items:
        if isinstance(v, dict):
            res = k + "_{" + params_to_name(v) + "}"
            name_list.append(res)
        else:
            name_list.append(k + "_" + str(v))
    return "_".join(name_list)


def summarize_moments(variable, name):
    mean, var = tf.nn.moments(variable, axes=[0])
    mean_sum = tf.scalar_summary("{} mean".format(name), mean[0])
    var_sum = tf.scalar_summary("{} variance".format(name), var[0])
    return [mean_sum, var_sum]


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


def parse_summary_ops(summary_str):
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary_str)
    summaries = {}
    for val in summary_proto.value:
        # Assuming all summaries are scalars.
        summaries[val.tag] = val.simple_value
    return summaries


def log(session, interval, step, summary_op, summary_writer, ops):
    if step % interval == 0:
        keys, values = zip(*ops.iteritems())
        outputs = session.run([summary_op] + list(ops.values()))
        summary_str = outputs[0]
        other_outputs = outputs[1:]

        output_dict = parse_summary_ops(summary_str)
        for key, output in zip(keys, other_outputs):
            output_dict[key] = output

        output = " ".join(["{}: {}".format(k,v) for (k,v) in output_dict.iteritems()])
        print(output)

        summary_writer.add_summary(summary_str, step)


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
