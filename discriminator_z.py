import tensorflow as tf
from utils import logger
import ops


class DiscriminatorZ(object):
    def __init__(self, name, is_train, norm='batch', activation='relu'):
        logger.info('Init DiscriminatorZ %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            D = input
            for i in range(3):
                D = ops.mlp(D, 512, 'FC512_{}'.format(i), self._is_train,
                            self._reuse, self._norm, self._activation)
            D = ops.mlp(D, 1, 'FC1_{}'.format(i), self._is_train,
                        self._reuse, norm=None, activation=None)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return D
