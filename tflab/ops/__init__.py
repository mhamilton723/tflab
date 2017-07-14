import tensorflow as tf
import os

#TODO add check for old_gcc
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

try:
    w2v = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
except:
    try:
        compile_ops(False)
        w2v = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
    except:
        compile_ops(True)
        w2v = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))


skipgram_word2vec = w2v.skipgram_word2vec
