import logging
import os


# start logging
logging.info("Start CycleGAN")
logger = logging.getLogger('cycle-gan')
logger.setLevel(logging.INFO)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
