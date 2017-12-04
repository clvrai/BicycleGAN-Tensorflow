import argparse
import sys
import signal
import os
from datetime import datetime

import tensorflow as tf

from data_loader import get_data
from model import BicycleGAN
from utils import logger, makedirs


# parsing cmd arguments
parser = argparse.ArgumentParser(description="Run commands")
def str2bool(v):
    return v.lower() == 'true'
parser.add_argument('--train', default=True, type=str2bool,
                    help="Training mode")
parser.add_argument('--task', type=str, default='edges2shoes',
                    help='Task name')
parser.add_argument('--coeff_kl', type=float, default=0.01,
                    help='Loss coefficient for KL divergence')
parser.add_argument('--coeff_reconstruct', type=float, default=10,
                    help='Loss coefficient for reconstruct')
parser.add_argument('--coeff_latent', type=float, default=0.5,
                    help='Loss coefficient for latent cycle')
parser.add_argument('--instance_normalization', default=False, type=bool,
                    help="Use instance norm instead of batch norm")
parser.add_argument('--log_step', default=100, type=int,
                    help="Tensorboard log frequency")
parser.add_argument('--batch_size', default=1, type=int,
                    help="Batch size")
parser.add_argument('--image_size', default=256, type=int,
                    help="Image size")
parser.add_argument('--latent_dim', default=8, type=int,
                    help="Dimensionality of latent vector")
parser.add_argument('--use_resnet', default=True, type=bool,
                    help="Use the ResNet model for the encoder")
parser.add_argument('--load_model', default='',
                    help='Model path to load (e.g., train_2017-07-07_01-23-45)')
parser.add_argument('--gpu', default="1", type=str,
                    help="gpu index for CUDA_VISIBLE_DEVICES")


class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args):
    # setting the GPU #
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.info('Read data:')
    train_A, train_B, test_A, test_B = get_data(args.task, args.image_size)

    logger.info('Build graph:')
    model = BicycleGAN(args)

    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    logger.info('Trainable vars:')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    if args.load_model != '':
        model_name = args.load_model
    else:
        model_name = '{}_{}'.format(args.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    logdir = './logs'
    makedirs(logdir)
    logdir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', logdir)
    summary_writer = tf.summary.FileWriter(logdir)

    makedirs('./results')

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=model.global_step,
                             save_model_secs=300,
                             save_summaries_secs=30)

    if args.train:
        logger.info("Starting training session.")
        with sv.managed_session() as sess:
            model.train(sess, summary_writer, train_A, train_B)

    logger.info("Starting testing session.")
    with sv.managed_session() as sess:
        base_dir = os.path.join('results', model_name)
        makedirs(base_dir)
        model.test(sess, test_A, test_B, base_dir)

def main():
    args, unparsed = parser.parse_known_args()

    def shutdown(signal, frame):
        tf.logging.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    run(args)

if __name__ == "__main__":
    main()
