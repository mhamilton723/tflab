import os
import warnings
from abc import ABCMeta, abstractmethod

import tensorflow as tf

from .utils import get_or_create_path
from tensorflow.core.framework import summary_pb2
from collections import defaultdict
import shutil
import pickle


class Serializable(object):
    saver = None

    def _create_saver(self):
        if self.saver is None:
            self.saver = tf.train.Saver()

    def save(self, save_path, name, session):
        self._create_saver()
        path = get_or_create_path(save_path, "checkpoints", name)
        self.saver.save(session, path)
        print("Saved to {}".format(path))

    def load(self, path, session):
        if os.path.exists(path + ".index"):
            print("Checkpoint Found: loading model from {}".format(path))
            self.saver.restore(session, path)
            return True
        else:
            warnings.warn("No Checkpoint found, starting from scratch", UserWarning)
            return False

    def load_or_initialize(self, save_path, name, session, try_load=True):
        self._create_saver()
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
        return k + "={" + str(v) + "}"

    def __init__(self, name, params, abbrev=None):
        Param.__init__(self, name, params, abbrev)


class Parametrizable(object):
    _param_map = {}

    def add_param(self, name, value, abbrev=None):
        self._param_map[name] = Param(name, value, abbrev)

    def add_params(self, params):
        if isinstance(params, dict):
            params = [Param(name, value) for name, value in params.iteritems()]
        for p in params:
            self._param_map[p.name] = p

    def add_param_group(self, name, params, abbrev=None):
        if isinstance(params, dict):
            params = [Param(k, v) for k, v in params.iteritems()]
        self._param_map[name] = ParamGroup(name, params, abbrev)

    def get_param(self, name):
        return self._param_map[name]

    def get_params(self):
        return self._param_map.values()

    def del_param(self, name):
        self._param_map.pop(name)

    def __str__(self):
        return "_".join(str(p) for p in sorted(self._param_map.values(), key=lambda p: p.abbrev_or_name()))


class Logger(object):
    _summaries = defaultdict(list)

    def summary_writer(self):
        raise NotImplementedError

    def emit_non_tensor(self, tag, value, step):
        val = summary_pb2.Summary.Value(tag=tag, simple_value=value)
        summary = summary_pb2.Summary(value=[val])
        self.summary_writer().add_summary(summary, step)
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
                additional_ops = {k: v for (k, v) in additional.iteritems() if isinstance(v, tf.Tensor)}
            else:
                additional_ops = {}

            outputs = session.run([tf.summary.merge(self._summaries[group])] + list(additional_ops.values()))
            summary_str = outputs[0]
            other_outputs = outputs[1:]

            output_dict = self._parse_summary_ops(summary_str)
            for key, output in zip(additional_ops.keys(), other_outputs):
                output_dict[key] = output

            terms = []
            for (k, v) in sorted(output_dict.iteritems()):
                if isinstance(v, float):
                    formatted = "%0.*e" % (precision, v)
                else:
                    formatted = v
                terms.append("{}: {}".format(k, formatted).ljust(len(str(k)) + precision + 9))

            output = " ".join(terms)
            print("Step: {} ".format(step).ljust(15) + output)
            self.summary_writer().add_summary(summary_str, step)


class Model(Serializable, Parametrizable, Logger):
    save_path = None

    def summary_path(self):
        return os.path.join(self.save_path, "summaries")

    def summary_writer(self):
        return tf.summary.FileWriter(
            get_or_create_path(self.summary_path(), str(self), exclude_last=False))

    def save(self, session):
        Serializable.save(self, self.save_path, str(self), session)

    def save_nontf(self, name, obj):
        path = get_or_create_path(self.save_path, "nontf", name, str(self))
        with open(path, "w+") as f:
            pickle.dump(obj, f)

    def load_nontf(self, name):
        path = os.path.join(self.save_path, "nontf", name, str(self))
        with open(path, 'r') as f:
            return pickle.load(f)

    def try_load_nontf(self, name, on_failure=None, try_load=True):
        path = os.path.join(self.save_path, "nontf", name, str(self))
        if os.path.exists(path) and try_load:
            print("Loading {} from file".format(name))
            return self.load_nontf(name)
        else:
            print("File {} does not exist".format(path))
            if on_failure is not None:
                print("Recomputing {}".format(name))
                return on_failure()

    def can_load(self):
        path = os.path.join(self.save_path, "checkpoints", str(self))
        return os.path.exists(path + ".index")

    def load_or_initialize(self, session, try_load=True, remove_old_logs=True):
        self._create_saver()
        tf.global_variables_initializer().run()
        if try_load:
            path = os.path.join(self.save_path, "checkpoints", str(self))
            wipe_logs = not self.load(path, session)
        else:
            wipe_logs = True

        if wipe_logs and remove_old_logs:
            old_logs = os.path.join(self.summary_path(), str(self))
            if os.path.isdir(old_logs):
                print("removing old logs at {}".format(old_logs))
                shutil.rmtree(old_logs)
