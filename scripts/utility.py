import logging
from pathlib import Path

import pandas as pd
import yaml


def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/

    Args:
        log_path [str]: eg: "../log/train.log"
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger
