import tensorflow as tf


# import theano.tensor as T

def mse(X, Y):
    return tf.reduce_mean(tf.squared_difference(X, Y))


# calculate the squared distance between x and y
def squared_cdistance(X_, Y_):
    r_ = tf.expand_dims(X_, 1, "r_1")
    intermediate = tf.square(r_ - Y_)
    result = tf.reduce_sum(intermediate, -1)
    return result


# apply a kernel to the squared distance
def gaussian(X_, Y_, squared_sigma):
    dists = squared_cdistance(X_, Y_)
    return tf.exp(-dists / squared_sigma)


def multiscale_gaussian(X_, Y_, squared_sigmas, weights):
    dists = squared_cdistance(X_, Y_)
    out = tf.zeros(tf.shape(dists))
    for scale, weight in zip(squared_sigmas, weights):
        out += weight * tf.exp(-dists / scale)
    return out


# define MMD cost
def mmd_squared(source, target, kernel):
    # MMD cost function
    print("Adding MMD calibration to the cost function")
    xx = kernel(source, source)
    # xx = tf.Print(xx, [tf.shape(xx)])
    #xx = tf.batch_matrix_set_diag(xx, tf.zeros((tf.shape(xx)[0],)))
    xy = kernel(source, target)
    yy = kernel(target, target)
    #yy = tf.batch_matrix_set_diag(yy, tf.zeros((tf.shape(yy)[0],)))
    MMD_squared = tf.reduce_mean(xx) - 2 * tf.reduce_mean(xy) + tf.reduce_mean(yy)
    #       return 0;
    return MMD_squared

def mmd(source,target, kernel):
    tf.sqrt(mmd_squared(source,target,kernel))
