import logging
import os


# start logging
logging.info("Start BicycleGAN")
logger = logging.getLogger('Bicycle-gan')
logger.setLevel(logging.INFO)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
