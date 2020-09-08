# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/log.py


import json
import os
import logging
from os.path import dirname, abspath, exists, join
from datetime import datetime



def make_run_name(format, framework, phase):
    return format.format(
        framework=framework,
        phase=phase,
        timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )


def make_logger(run_name, log_output):
    if log_output is not None:
        run_name = log_output.split('/')[-1].split('.')[0]
    logger = logging.getLogger(run_name)
    logger.propagate = False
    log_filepath = log_output if log_output is not None else join('logs', f'{run_name}.log')

    log_dir = dirname(abspath(log_filepath))
    if not exists(log_dir):
        os.makedirs(log_dir)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger


def make_checkpoint_dir(checkpoint_dir, run_name):
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else join('checkpoints', run_name)
    if not exists(abspath(checkpoint_dir)):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir
