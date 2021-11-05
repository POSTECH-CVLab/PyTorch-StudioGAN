# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/log.py

from os.path import dirname, exists, join
from datetime import datetime
import json
import os
import logging


def make_run_name(format, data_name, framework, phase):
    return format.format(data_name=data_name,
                         framework=framework,
                         phase=phase,
                         timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))


def make_logger(save_dir, run_name, log_output):
    if log_output is not None:
        run_name = log_output.split('/')[-1].split('.')[0]
    logger = logging.getLogger(run_name)
    logger.propagate = False
    log_filepath = log_output if log_output is not None else join(save_dir, "logs", run_name + ".log")

    log_dir = dirname(log_filepath)
    if not exists(log_dir):
        os.makedirs(log_dir)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger
