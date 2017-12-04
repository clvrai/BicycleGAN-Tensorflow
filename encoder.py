import tensorflow as tf
from utils import logger
import ops


class Encoder(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky',
                 image_size=128, latent_dim=8, use_resnet=True):
        logger.info('Init Encoder %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._image_size = image_size
        self._latent_dim = latent_dim
        self._use_resnet = use_resnet

    def __call__(self, input):
        if self._use_resnet:
            return self._resnet(input)
        else:
            return self._convnet(input)

    def _convnet(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            num_filters = [64, 128, 256, 512, 512, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            E = input
            for i, n in enumerate(num_filters):
                E = ops.conv_block(E, n, 'C{}_{}'.format(n, i), 4, 2, self._is_train,
                                   self._reuse, norm=self._norm if i else None, activation='leaky')
            E = ops.flatten(E)
            mu = ops.mlp(E, self._latent_dim, 'FC8_mu', self._is_train, self._reuse,
                         norm=None, activation=None)
            log_sigma = ops.mlp(E, self._latent_dim, 'FC8_sigma', self._is_train, self._reuse,
                                norm=None, activation=None)

            z = mu + tf.random_normal(shape=tf.shape(self._latent_dim)) * tf.exp(log_sigma)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return z, mu, log_sigma

    def _resnet(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            num_filters = [128, 256, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            E = input
            E = ops.conv_block(E, 64, 'C{}_{}'.format(64, 0), 4, 2, self._is_train,
                               self._reuse, norm=None, activation='leaky', bias=True)
            for i, n in enumerate(num_filters):
                E = ops.residual(E, n, 'res{}_{}'.format(n, i + 1), self._is_train,
                                 self._reuse, norm=self._norm, activation='leaky',
                                 bias=True)
                E = tf.nn.avg_pool(E, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            E = tf.nn.avg_pool(E, [1, 8, 8, 1], [1, 8, 8, 1], 'SAME')
            E = ops.flatten(E)
            mu = ops.mlp(E, self._latent_dim, 'FC8_mu', self._is_train, self._reuse,
                         norm=None, activation=None)
            log_sigma = ops.mlp(E, self._latent_dim, 'FC8_sigma', self._is_train, self._reuse,
                                norm=None, activation=None)

            z = mu + tf.random_normal(shape=tf.shape(self._latent_dim)) * tf.exp(log_sigma)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return z, mu, log_sigma
