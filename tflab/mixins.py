import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import os
import warnings
from collections import defaultdict
import shutil
import pickle
import numpy as np
import h5py

from scipy.sparse import dok_matrix
from .utils import get_or_create_path
from six.moves import xrange


class Serializable(object):

    def __init__(self):
        self._vars = set()
        self._saver = None

    def _update_vars(self):
        self._vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def saver(self):
        old_vars = self._vars
        self._update_vars()

        if len(self._vars)==0 and self._saver is None:
            raise ValueError("Need to have variables to save before calling the saver")

        if old_vars != self._vars:
            self._saver = tf.train.Saver()
        return self._saver

    def save(self, save_path, name, session):
        path = get_or_create_path(save_path, "checkpoints", name)
        self.saver().save(session, path)
        print("Saved to {}".format(path))

    def load(self, path, session):
        if os.path.exists(path + ".index"):
            print("Checkpoint Found: loading model from {}".format(path))
            self.saver().restore(session, path)
            return True
        else:
            warnings.warn("No Checkpoint found, starting from scratch", UserWarning)
            return False

    def load_or_initialize(self, save_path, name, session, try_load=True):
        tf.global_variables_initializer().run()
        if try_load:
            path = os.path.join(save_path, "checkpoints", name)
            self.load(path, session)


class Param(object):
    def abbrev_or_name(self):
        if self.abbrev is not None:
            return self.abbrev
        else:
            return self.name

    def __str__(self):
        k = self.abbrev_or_name()
        v = self.value_to_string(self.value)
        return "{}={}".format(k, v)

    def __init__(self, name, value, abbrev=None, value_to_string=lambda v: str(v)):
        self.name = name
        self.value = value
        self.abbrev = abbrev
        self.value_to_string = value_to_string


class ParamGroup(Param):
    def __str__(self):
        k = self.abbrev_or_name()
        v = '_'.join(str(param) for param in
                     sorted(self.value, key=lambda p: p.abbrev_or_name()))
        return k + "={" + v + "}"

    def __init__(self, name, params, abbrev=None):
        Param.__init__(self, name, params, abbrev)


class Parametrizable(object):
    def __init__(self):
        self._param_map = {}

    def add_param(self, name, value, abbrev=None):
        self._param_map[name] = Param(name, value, abbrev)

    def add_params(self, params):
        if isinstance(params, dict):
            params = [Param(name, value) for name, value in params.items()]
        for p in params:
            self._param_map[p.name] = p

    def add_param_group(self, name, params, abbrev=None):
        if name in self._param_map:
            raise ValueError("The param map already contains a parameter with name " + name)
        if isinstance(params, dict):
            params = [Param(k, v) for k, v in params.items()]
        self._param_map[name] = ParamGroup(name, params, abbrev)

    def get_param(self, name):
        return self._param_map[name]

    def get_params(self):
        return list(self._param_map.values())

    def del_param(self, name):
        self._param_map.pop(name)

    def __str__(self):
        return "_".join(str(p) for p in sorted(list(self._param_map.values()), key=lambda p: p.abbrev_or_name()))


class Logger(object):

    def __init__(self):
        self._summaries = defaultdict(list)

    def summary_writer(self):
        raise NotImplementedError

    def emit_non_tensor(self, tag, value, step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer().add_summary(summary, step)
        self.summary_writer().flush()
        print("Step: {} {}: {}".format(step, tag, value))

    def _summarize_moments(self, name, variable):
        mean, var = tf.nn.moments(variable, axes=[0])
        mean_sum = tf.summary.scalar("{} mean".format(name), mean[0])
        var_sum = tf.summary.scalar("{} variance".format(name), var[0])
        return [mean_sum, var_sum]

    def _parse_summary_ops(self, summary_str):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summary_str)
        summaries = {}
        for val in summary_proto.value:
            # Assuming all summaries are scalars.
            summaries[val.tag] = val.simple_value
        return summaries

    def log(self, summary, group=None):
        self._summaries[group].append(summary)

    def remove_log_group(self, group=None):
        self._summaries.pop(group)

    def log_scalar(self, name, tensor, group=None):
        self.log(tf.summary.scalar(name, tensor), group)

    def log_moments(self, name, tensor, group=None):
        mean, variance = self._summarize_moments(name, tensor)
        self.log(mean, group)
        self.log(variance, group)

    def log_histogram(self, name, tensor, group=None):
        self.log(tf.summary.histogram(name, tensor), group)

    def log_image(self, name, tensor, group=None, no_minibatch=False, no_channels=False):

        if no_minibatch:
            tensor = tf.expand_dims(tensor, 0)
        if no_channels:
            tensor = tf.expand_dims(tensor, -1)

        self.log(tf.summary.image(name, tensor), group)

    def log_emitter(self, session, step, interval=1000, group=None, additional=None, precision=4):
        if step % interval == 0:
            if additional is not None:
                additional_ops = {k: v for (k, v) in additional.items() if isinstance(v, tf.Tensor)}
            else:
                additional_ops = {}

            outputs = session.run([tf.summary.merge(self._summaries[group])] + list(additional_ops.values()))
            summary_str = outputs[0]
            other_outputs = outputs[1:]

            output_dict = self._parse_summary_ops(summary_str)
            for key, output in zip(additional_ops.keys(), other_outputs):
                output_dict[key] = output

            terms = []
            for (k, v) in sorted(output_dict.items()):
                if isinstance(v, float):
                    formatted = "%0.*e" % (precision, v)
                else:
                    formatted = v
                terms.append("{}: {}".format(k, formatted).ljust(len(str(k)) + precision + 9))

            output = " ".join(terms)
            print("Step: {} ".format(step).ljust(15) + output)
            self.summary_writer().add_summary(summary_str, step)


class Model(Serializable, Parametrizable, Logger):

    def __init__(self, save_path=None):
        self.save_path = save_path
        self._summary_writer = None
        self._summary_path = None
        Parametrizable.__init__(self)
        Logger.__init__(self)
        Serializable.__init__(self)

    def get_save_path(self):
        if self.save_path is None:
            raise AttributeError("need to define a save path before using the save path")
        else:
            return self.save_path

    def summary_path(self):
        return os.path.join(self.get_save_path(), "summaries")

    def summary_writer(self):
        if self._summary_writer is None or self._summary_path != self.summary_path():
            self._summary_writer = tf.summary.FileWriter(
                get_or_create_path(self.summary_path(), str(self), exclude_last=False))
            self._summary_path = os.path.join(self.summary_path(), str(self))

        return self._summary_writer

    def save(self, session):
        Serializable.save(self, self.get_save_path(), str(self), session)

    def save_nontf(self, name, *objs):
        path = get_or_create_path(self.get_save_path(), "nontf", name, str(self), exclude_last=False)
        for dir in os.listdir(path):
            f = os.path.join(path, dir)
            try:
                shutil.rmtree(f)
            except NotADirectoryError:
                os.remove(f)
        print("Saving to {}".format(path))
        for i, obj in enumerate(objs):
            if isinstance(obj, (np.ndarray, np.generic)) and np.issubdtype(obj.dtype, np.number):
                print("saving obj {} as numpy".format(i))
                key = "numpy"
                sub_filename = os.path.join(path, "{}_{}_{}".format(i, key, ".hdf5"))
                with h5py.File(sub_filename, 'w') as hf:
                    hf.create_dataset('data', data=obj)

            elif isinstance(obj, dok_matrix):
                print("saving obj {} as dok".format(i))
                key = "dok"
                sub_filename = os.path.join(path, "{}_{}_{}".format(i, key, ".hdf5"))
                keys, values = zip(*obj.items())
                with h5py.File(sub_filename, 'w') as hf:
                    hf.create_dataset('keys', data=np.array(keys))
                    hf.create_dataset('values', data=np.array(values), dtype=obj.dtype)
                    hf.create_dataset('shape', data=np.array(obj.shape))

            else:
                print("saving obj {} as pkl".format(i))
                key = "pkl"
                sub_filename = os.path.join(path, "{}_{}_{}".format(i, key, ".pkl"))
                with open(sub_filename, "wb+") as f:
                    pickle.dump(obj, f)

    def load_nontf(self, name):
        path = os.path.join(self.get_save_path(), "nontf", name, str(self))
        sub_filenames = [os.path.join(path, fn) for fn in os.listdir(path)]
        data_dict = {}
        print("Loading from {}".format(path))

        for sub_filename in sub_filenames:
            number, key, extension = os.path.basename(sub_filename).split("_")
            if key == "numpy":
                with h5py.File(sub_filename, 'r') as hf:
                    data_dict[number] = np.array(hf.get('data'))
            elif key == "dok":
                with h5py.File(sub_filename, 'r') as hf:
                    keys = hf.get('keys')
                    values = hf.get('values')
                    shape = np.array(hf.get('shape'))
                    dok = dok_matrix(tuple(shape), dtype=values.dtype)
                    print("creating dok")
                    for k, v in zip(keys, values):
                        dok[k] = v
                    print("done creating dok")
                    del keys
                    del values
                    data_dict[number] = dok
            elif key == "pkl":
                with open(sub_filename, 'rb') as f:
                    data_dict[number] = pickle.load(f)
        objs = [p[1] for p in sorted(data_dict.items())]
        return objs

    def try_load_nontf(self, name, on_failure=None, try_load=True):
        path = os.path.join(self.get_save_path(), "nontf", name, str(self))
        if os.path.exists(path) and try_load:
            print("Loading {} from file".format(name))
            return self.load_nontf(name)
        else:
            if not os.path.exists(path):
                print("File {} does not exist".format(path))
            if on_failure is not None:
                print("Recomputing {}".format(name))
                return on_failure()

    def can_load(self):
        path = os.path.join(self.get_save_path(), "checkpoints", str(self))
        return os.path.exists(path + ".index")

    def load_or_initialize(self, session, try_load=True, remove_old_logs=True):
        tf.global_variables_initializer().run()
        if try_load:
            path = os.path.join(self.get_save_path(), "checkpoints", str(self))
            wipe_logs = not self.load(path, session)
        else:
            wipe_logs = True

        if wipe_logs and remove_old_logs:
            old_logs = os.path.join(self.summary_path(), str(self))
            if os.path.isdir(old_logs):
                print("removing old logs at {}".format(old_logs))
                shutil.rmtree(old_logs)
            else:
                print("{} is not a directory".format(old_logs))
