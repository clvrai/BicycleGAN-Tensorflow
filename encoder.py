import tensorflow as tf
from utils import logger
import ops


class Encoder(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky',
                 image_size=128, latent_dim=8):
        logger.info('Init Encoder %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._image_size = image_size
        self._latent_dim = latent_dim

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            num_filters = [64, 128, 256, 512, 512, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            E = input
            for i, n in enumerate(num_filters):
                E = ops.conv_block(E, n, 'C{}_{}'.format(n, i), 4, 2, self._is_train,
                                self._reuse, norm=self._norm if i else None, activation='leaky')
            E = tf.reshape(E, [-1, 512])
            E = ops.mlp(E, self._latent_dim, 'FC8', self._is_train, self._reuse,
                        norm=None, activation=None)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return E
