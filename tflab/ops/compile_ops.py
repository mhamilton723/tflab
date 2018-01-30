import os
import sys
import tensorflow as tf
from distutils.version import StrictVersion as sv

#TODO add check for old_gcc
def compile_ops(old_gcc=True):
    start_dir = os.curdir
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    files = "word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so"
    ABI_compat = "" if old_gcc else "-D_GLIBCXX_USE_CXX11_ABI=0"

    if (sv(tf.__version__) >= "1.5.0"):
        inc_flags = " ".join(tf.sysconfig.get_compile_flags())
        lib_flags = " ".join(tf.sysconfig.get_link_flags())
        # In 1.5.0, inc_flags sets the ABI flag to 1
        if ("-D_GLIBCXX_USE_CXX11_ABI" in inc_flags):
            ABI_compat = ""
    elif (sv(tf.__version__) >= "1.4.0" and sv(tf.__version__) < "1.5.0"):
        tf_inc = tf.sysconfig.get_include()
        tf_lib = tf.sysconfig.get_lib()
        inc_flags = "-I{0} -I{0}/external/nsync/public".format(tf_inc)
        lib_flags = "-L{} -ltensorflow_framework".format(tf_lib)
    else:
        inc_flags = "-I{}".format(tf.sysconfig.get_include())
        lib_flags = ""
    inc_lib = "{} {}".format(inc_flags, lib_flags)

    os.system("g++ -std=c++11 -shared -O2 -fPIC "
              "{} {} {}".format(ABI_compat, files, inc_lib))

    os.chdir(start_dir)
