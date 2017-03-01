from lib.utils import get_or_create_path
from abc import ABCMeta
import os
import warnings
import tensorflow as tf


class Serializable:
    __metaclass__ = ABCMeta

    @property
    def saver(self):
        return tf.train.Saver()

    def save_model(self, save_path, name, session):
        path = get_or_create_path(save_path, "checkpoints", "checkpoints", name + ".ckpt")
        self.saver.save(session, path)

    def load_model(self, save_path, name, session):
        path = os.path.join(save_path, "checkpoints", "checkpoints", name + ".ckpt")
        if os.path.exists(path):
            print("Checkpoint Found: loading model from {}".format(path))
            self.saver.restore(session, path)
        else:
            warnings.warn("No Checkpoint found, starting from scratch", UserWarning)
