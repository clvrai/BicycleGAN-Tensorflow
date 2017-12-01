import tensorflow as tf
from utils import logger
import ops


class Generator(object):
    def __init__(self, name, is_train, norm='batch', image_size=128):
        logger.info('Init Generator %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._reuse = False
        self._image_size = image_size

    def __call__(self, input, z):
        with tf.variable_scope(self.name, reuse=self._reuse):
            batch_size = int(input.get_shape()[0])
            latent_dim = int(z.get_shape()[-1])
            num_filters = [64, 128, 256, 512, 512, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            layers = []
            G = input
            z = tf.reshape(z, [batch_size, 1, 1, latent_dim])
            z = tf.tile(z, [1, self._image_size, self._image_size, 1])
            G = tf.concat([G, z], axis=3)
            for i, n in enumerate(num_filters):
                G = ops.conv_block(G, n, 'C{}_{}'.format(n, i), 4, 2, self._is_train,
                                self._reuse, norm=self._norm if i else None, activation='leaky')
                layers.append(G)

            layers.pop()
            num_filters.pop()
            num_filters.reverse()

            for i, n in enumerate(num_filters):
                G = ops.deconv_block(G, n, 'CD{}_{}'.format(n, i), 4, 2, self._is_train,
                                self._reuse, norm=self._norm, activation='relu')
                G = tf.concat([G, layers.pop()], axis=3)
            G = ops.deconv_block(G, 3, 'last_layer', 4, 2, self._is_train,
                               self._reuse, norm=None, activation='tanh')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G
