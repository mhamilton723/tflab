import os
import sys
import tensorflow as tf


def compile_ops(old_gcc=True):
    start_dir = os.curdir
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.environ["TF_INC"] = tf.sysconfig.get_include()
    if old_gcc:
        fill = ""
    else:
        fill = "-D_GLIBCXX_USE_CXX11_ABI=0"
    os.system("g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc "
              "-o word2vec_ops.so -fPIC -I $TF_INC -O2 {}".format(fill))
    os.chdir(start_dir)