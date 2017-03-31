import os
import warnings
from abc import ABCMeta, abstractproperty

import tensorflow as tf

from .utils import get_or_create_path
from tensorflow.core.framework import summary_pb2
from collections import defaultdict


class Serializable:
    __metaclass__ = ABCMeta

    @property
    def saver(self):
        return tf.train.Saver()

    def save_model(self, save_path, name, session):
        path = get_or_create_path(save_path, "checkpoints", name)
        self.saver.save(session, path)
        print("Saved to {}".format(path))

    def load_model(self, save_path, name, session, initialize):
        path = os.path.join(save_path, "checkpoints", name)
        if os.path.exists(path + ".index"):
            print("Checkpoint Found: loading model from {}".format(path))
            self.saver.restore(session, path)
        else:
            warnings.warn("No Checkpoint found, starting from scratch", UserWarning)

            if initialize:
                tf.global_variables_initializer().run()


class Logger:
    __metaclass__ = ABCMeta

    _summaries = defaultdict(list)

    @abstractproperty
    def summary_writer(self):
        pass

    def emit_non_tensor(self, tag, value, step):
        val = summary_pb2.Summary.Value(tag=tag, simple_value=value)
        summary = summary_pb2.Summary(value=[val])
        self.summary_writer.add_summary(summary, step)
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

    def log_scalar(self, name, tensor, group=None):
        self.log(tf.summary.scalar(name, tensor), group)

    def log_moments(self, name, tensor, group=None):
        mean, variance = self._summarize_moments(name, tensor)
        self.log(mean, group)
        self.log(variance, group)

    def log_histogram(self, name, tensor, group=None):
        self.log(tf.summary.histogram(name, tensor), group)

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
            self.summary_writer.add_summary(summary_str, step)
