from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
import tensorflow as tf


class ASGradientDescentOptimizer(optimizer.Optimizer):
    """Optimizer that implements the gradient descent algorithm.

    @@__init__
    """

    def __init__(self, base_learning_rate, scale=1.01, use_locking=False, name="GradientDescent"):
        """Construct a new gradient descent optimizer.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescent".
        """
        super(ASGradientDescentOptimizer, self).__init__(use_locking, name)
        self._base_learning_rate = base_learning_rate
        self._scale = scale

        self._learning_rate_tensor = None
        self._scale_tensor = None
        self._previous_grad_tensor = None

    def _create_slots(self, var_list):
        for v in var_list:
            lr_val = constant_op.constant(self._base_learning_rate, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, lr_val, "learning_rate", self._name)
            self._zeros_slot(v, "previous_grad", self._name)

    def _prepare(self):
        self._scale_tensor = ops.convert_to_tensor(self._scale, name="scale")

    def _apply_dense(self, grad, var):
        previous_grad = self.get_slot(var, "previous_grad")
        lr = self.get_slot(var, "learning_rate")
        scale_factor = tf.pow(self._scale_tensor, tf.sign(grad * previous_grad))
        lr_update = lr.assign(lr * scale_factor)
        with tf.control_dependencies([lr_update]):
            previous_grad_update = previous_grad.assign(grad)
            with tf.control_dependencies([previous_grad_update]):
                apply_grad_op = training_ops.apply_gradient_descent(
                    var, 1.0, lr*grad, use_locking=self._use_locking).op

        return apply_grad_op

    def _apply_sparse(self, grad, var):
        #previous_grad = self.get_slot(var, "previous_grad")
        #lr = self.get_slot(var, "learning_rate")

        #scale_factor = tf.pow(self._scale_tensor, tf.sign(grad * previous_grad))
        #lr_update = lr.assign(lr * scale_factor)

        #delta = ops.IndexedSlices(
        #    grad.values *
        #    math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        #    grad.indices, grad.dense_shape)
        #return var.scatter_sub(delta, use_locking=self._use_locking)
        raise NotImplementedError()




class ASRMSPropOptimizer(optimizer.Optimizer):
    """Optimizer that implements the RMSProp algorithm.

    See the [paper]
    (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

    @@__init__
    """

    def __init__(self,
                 base_learning_rate,
                 scale=1.01,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 use_locking=False,
                 name="ASRMSProp"):
        """Construct a new RMSProp optimizer.

        Note that in dense implement of this algorithm, m_t and v_t will
        update even if g is zero, but in sparse implement, m_t and v_t
        will not update in iterations g is zero.

        Args:
          learning_rate: A floating point value.  The learning rate.
          decay: Discounting factor for the history/coming gradient
          momentum: A scalar tensor.
          epsilon: Small value to avoid zero denominator.
          use_locking: If True use locks for update operation.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "RMSProp".
        """
        super(ASRMSPropOptimizer, self).__init__(use_locking, name)
        self._base_learning_rate = base_learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._scale = scale

        # Tensors for learning rate and momentum.  Created in _prepare.
        self._learning_rate_tensor = None
        self._scale_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None
        self._previous_grad_tensor = None

    def _create_slots(self, var_list):
        for v in var_list:
            rms_val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, rms_val, "rms", self._name)

            lr_val = constant_op.constant(self._base_learning_rate, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, lr_val, "learning_rate", self._name)

            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "previous_grad", self._name)

    def _prepare(self):
        self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
        self._scale_tensor = ops.convert_to_tensor(self._scale, name="scale")
        self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                                      name="momentum")
        self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                                     name="epsilon")

    def _apply_rms_prop(self, var, ms, mom, lr, rho, momentum, epsilon, grad):
        ms_op = ms.assign(rho * ms + (1 - rho) * grad * grad)
        with tf.control_dependencies([ms_op]):
            mom_op = mom.assign(momentum * mom + lr * grad / tf.sqrt(ms + epsilon))
            with tf.control_dependencies([mom_op]):
                var_op = var.assign(var - mom)
        return var_op

    def _apply_dense(self, grad, var):
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        previous_grad = self.get_slot(var, "previous_grad")

        lr = self.get_slot(var, "learning_rate")
        scale_factor = tf.pow(self._scale_tensor, tf.sign(grad * previous_grad))
        lr_update = lr.assign(lr * scale_factor)

        with tf.control_dependencies([lr_update]):
            previous_grad_update = previous_grad.assign(grad)
            with tf.control_dependencies([previous_grad_update]):
                rms_prop_op = self._apply_rms_prop(
                    var, rms, mom, lr,
                    math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
                    math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
                    math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
                    grad).op

        return rms_prop_op

    def _apply_sparse(self, grad, var):
        '''
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        return training_ops.sparse_apply_rms_prop(
            var, rms, mom,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
            math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            use_locking=self._use_locking)
        '''
        raise NotImplementedError
