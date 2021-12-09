# -*- coding: utf-8 -*-

"""
This script is used to run convert data raw data to train and test data
It is designed to be idempotent [stateless transformation]

Usage:
    python3 ./scripts/etl.py
"""

import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yaml")
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

    ##################
    # Data transformation
    ##################
    logger.info("--------------------Start data transformation-------------------")
    df_campus = pd.read_csv(raw_data_file)

    df_campus["status"].replace("Placed", 1, inplace=True)
    df_campus["status"].replace("Not Placed", 0, inplace=True)

    # Binning data
    df_campus["ssc_p_bin"] = pd.qcut(df_campus["ssc_p"], 8)
    df_campus["hsc_p_bin"] = pd.qcut(df_campus["hsc_p"], 4)
    df_campus["degree_p_bin"] = pd.qcut(df_campus["degree_p"], 6)
    df_campus["etest_p_bin"] = pd.qcut(df_campus["etest_p"], 8)
    df_campus["mba_p_bin"] = pd.qcut(df_campus["mba_p"], 10)

    # Label Encoding
    non_numeric_features = [
        "gender",
        "ssc_b",
        "hsc_b",
        "hsc_s",
        "degree_t",
        "workex",
        "specialisation",
        "ssc_p_bin",
        "hsc_p_bin",
        "degree_p_bin",
        "etest_p_bin",
        "mba_p_bin",
    ]

    for feature in non_numeric_features:
        df_campus[feature] = LabelEncoder().fit_transform(df_campus[feature])

    # One-Hot Encoding
    categorical_features = ["ssc_b", "hsc_b", "hsc_s", "degree_t"]

    encoded_features = []

    for feature in categorical_features:
        encoded_feat = (
            OneHotEncoder()
            .fit_transform(df_campus[feature].values.reshape(-1, 1))
            .toarray()
        )
        n = df_campus[feature].nunique()
        cols = ["{}_{}".format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df_campus.index
        encoded_features.append(encoded_df)

    df_campus = pd.concat([df_campus, *encoded_features], axis=1)

    # Droping unnecessary columns
    drop_cols = [
        "sl_no",
        "ssc_p",
        "ssc_b",
        "hsc_p",
        "hsc_b",
        "hsc_s",
        "degree_p",
        "degree_t",
        "etest_p",
        "mba_p",
        "salary",
    ]

    df_campus.drop(columns=drop_cols, inplace=True)

    ##################
    # Train test split & Export
    ##################
    logger.info("------------------Train test split & Export------------------")
    train, test = train_test_split(df_campus, test_size=test_size)

    # Export data
    logger.info(f"Write data do {processed_path}")
    train.to_csv(processed_path / "train.csv", index=False)
    test.to_csv(processed_path / "test.csv", index=False)
    logger.info(f"Train: {train.shape}")
    logger.info(f"Test: {test.shape}")
    logger.info(f"\n")


if __name__ == "__main__":
    etl()
