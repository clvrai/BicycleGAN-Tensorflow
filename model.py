import os
import random

from tqdm import trange, tqdm
from scipy.misc import imsave
import tensorflow as tf
import numpy as np

from generator import Generator
from encoder import Encoder
from discriminator import Discriminator
from discriminator_z import DiscriminatorZ
from utils import logger


class BicycleGAN(object):
    def __init__(self, args):
        self._log_step = args.log_step
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._latent_dim = args.latent_dim
        self._lambda1 = args.lambda1
        self._lambda2 = args.lambda2
        self._gamma = args.gamma

        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._image_shape = [self._image_size, self._image_size, 3]

        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        image_a = self.image_a = \
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_a')
        image_b = self.image_b = \
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_b')
        z = self.z = \
            tf.placeholder(tf.float32, [self._batch_size, self._latent_dim], name='z')

        # Data augmentation
        seed = random.randint(0, 2**31 - 1)
        def augment_image(image):
            image = tf.image.resize_images(image, [self._augment_size, self._augment_size])
            image = tf.random_crop(image, [self._batch_size] + self._image_shape, seed=seed)
            image = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, seed), image)
            return image

        image_a = tf.cond(self.is_train,
                          lambda: augment_image(image_a),
                          lambda: image_a)
        image_b = tf.cond(self.is_train,
                          lambda: augment_image(image_b),
                          lambda: image_b)

        # Generator
        G = Generator('G', is_train=self.is_train,
                      norm='batch', image_size=self._image_size)

        # Discriminator
        D = Discriminator('D', is_train=self.is_train,
                          norm='batch', activation='leaky',
                          image_size=self._image_size)
        Dz = DiscriminatorZ('Dz', is_train=self.is_train,
                             norm='batch', activation='relu')

        # Encoder
        E = Encoder('E', is_train=self.is_train,
                    norm='batch', activation='relu',
                    image_size=self._image_size, latent_dim=self._latent_dim)

        # Generate images (a->b)
        image_ab = self.image_ab = G(image_a, z)
        z_reconstruct = E(image_ab)

        # Encode z (G(A, z) -> z)
        z_encoded = E(image_b)
        image_ab_encoded = G(image_a, z_encoded)

        # Discriminate real/fake images
        D_real = D(image_b)
        D_fake = D(image_ab)
        D_fake_encoded = D(image_ab_encoded)
        Dz_real = Dz(z)
        Dz_fake = Dz(z_encoded)

        loss_image_reconstruct = tf.reduce_mean(tf.abs(image_b - image_ab_encoded))

        loss_gan = (tf.reduce_mean(tf.squared_difference(D_real, 0.9)) +
            tf.reduce_mean(tf.square(D_fake))) * 0.5

        loss_image_cycle = (tf.reduce_mean(tf.squared_difference(D_real, 0.9)) +
            tf.reduce_mean(tf.square(D_fake_encoded))) * 0.5

        loss_latent_cycle = tf.reduce_mean(tf.abs(z - z_reconstruct))

        loss_Dz = (tf.reduce_mean(tf.squared_difference(Dz_real, 0.9)) +
            tf.reduce_mean(tf.square(Dz_fake))) * 0.5

        loss = self._gamma * loss_Dz \
            + loss_image_cycle - self._lambda1 * loss_image_reconstruct \
            + loss_gan - self._lambda2 * loss_latent_cycle

        # Optimizer
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss, var_list=D.var_list, global_step=self.global_step)
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(-loss, var_list=G.var_list)
        self.optimizer_Dz = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss, var_list=Dz.var_list)
        self.optimizer_E = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(-loss, var_list=E.var_list)

        # Summaries
        self.loss_image_reconstruct = loss_image_reconstruct
        self.loss_image_cycle = loss_image_cycle
        self.loss_latent_cycle = loss_latent_cycle
        self.loss_gan = loss_gan
        self.loss_Dz = loss_Dz
        self.loss = loss

        tf.summary.scalar('loss/image_reconstruct', loss_image_reconstruct)
        tf.summary.scalar('loss/image_cycle', loss_image_cycle)
        tf.summary.scalar('loss/latent_cycle', loss_latent_cycle)
        tf.summary.scalar('loss/gan', loss_gan)
        tf.summary.scalar('loss/Dz', loss_Dz)
        tf.summary.scalar('loss/total', loss)
        tf.summary.scalar('model/D_real', tf.reduce_mean(D_real))
        tf.summary.scalar('model/D_fake', tf.reduce_mean(D_fake))
        tf.summary.scalar('model/D_fake_encoded', tf.reduce_mean(D_fake_encoded))
        tf.summary.scalar('model/Dz_real', tf.reduce_mean(Dz_real))
        tf.summary.scalar('model/Dz_fake', tf.reduce_mean(Dz_fake))
        tf.summary.scalar('model/lr', self.lr)
        tf.summary.image('image/A', image_a[0:1])
        tf.summary.image('image/B', image_b[0:1])
        tf.summary.image('image/A-B', image_ab[0:1])
        tf.summary.image('image/A-B_encoded', image_ab_encoded[0:1])
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')
        logger.info('  {} images from A'.format(len(data_A)))
        logger.info('  {} images from B'.format(len(data_B)))

        data_size = min(len(data_A), len(data_B))
        num_batch = data_size // self._batch_size
        epoch_length = num_batch * self._batch_size

        num_initial_iter = 8
        num_decay_iter = 2
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter

        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:
            #TODO: resume training with global_step
            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)

            if iter == 0:
                data = zip(data_A, data_B)
                random.shuffle(data)
                data_A, data_B = zip(*data)

            image_a = np.stack(data_A[iter*self._batch_size:(iter+1)*self._batch_size])
            image_b = np.stack(data_B[iter*self._batch_size:(iter+1)*self._batch_size])
            sample_z = np.random.uniform(-1, 1, size=(self._batch_size, self._latent_dim))

            fetches = [self.loss,
                       self.optimizer_D, self.optimizer_Dz,
                       self.optimizer_G, self.optimizer_E]
            if step % self._log_step == 0:
                fetches += [self.summary_op]

            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.z: sample_z})

            z = np.random.uniform(-1, 1, size=(1, self._latent_dim))
            image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.lr: lr,
                                                    self.z: z,
                                                    self.is_train: True})
            imsave('results/r_{}.jpg'.format(step), np.squeeze(image_ab, axis=0))

            if step % self._log_step == 0:
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                t.set_description('Loss({:.3f})'.format(fetched[0]))

    def test(self, sess, data_A, data_B, base_dir):
        step = 0
        for (dataA, dataB) in tqdm(zip(data_A, data_B)):
            step += 1
            image_a = np.expand_dims(dataA, axis=0)
            image_b = np.expand_dims(dataB, axis=0)
            images = []
            images.append(image_a)
            images.append(image_b)

            for i in range(23):
                z = np.random.uniform(-1, 1, size=(1, self._latent_dim))
                image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                        self.z: z,
                                                        self.is_train: True})
                images.append(image_ab)

            image_rows = []
            for i in range(5):
                image_rows.append(np.concatenate(images[i*5:(i+1)*5], axis=2))
            images = np.concatenate(image_rows, axis=1)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'random_{}.jpg'.format(step)), images)

        step=0
        for (dataA, dataB) in tqdm(zip(data_A, data_B)):
            step += 1
            image_a = np.expand_dims(dataA, axis=0)
            image_b = np.expand_dims(dataB, axis=0)
            images = []
            images.append(image_a)
            images.append(image_b)

            for i in range(23):
                z = np.zeros((1, self._latent_dim))
                z[0][0] = (i / 23.0 - 0.5) * 2.0
                image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                        self.z: z,
                                                        self.is_train: True})
                images.append(image_ab)

            image_rows = []
            for i in range(5):
                image_rows.append(np.concatenate(images[i*5:(i+1)*5], axis=2))
            images = np.concatenate(image_rows, axis=1)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'linear_{}.jpg'.format(step)), images)
