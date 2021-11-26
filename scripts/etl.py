import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utility import parser_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yml")
def etl(config_file):
    """
    ETL function load raw data and convert to train and test set

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/etl.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    raw_data_file = config["etl"]["raw_data_file"]
    processed_path = Path(config["etl"]["processed_path"])
    test_size = config["etl"]["test_size"]
    logger.info(f"config: {config['etl']}")
