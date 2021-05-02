# -*- coding: utf-8 -*-
import logging
import pandas as pd
import click
import yaml
from dotmap import DotMap
from pathlib import Path
from transform_data import read_params
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--config_path', help="Path to Configuration File")
def split_dataset(config_path: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger()
    logger.info('Parsing the config file supplied')
    config = read_params(config_path)
    
    logger.info('Reading raw data...')
    
    raw_data_path = Path.cwd()/config.load_data['raw_dataset_path']/config.data_source['filename']
    raw_df = pd.read_csv(raw_data_path)
    
    logger.info('Splitting raw data...')
    train, test = train_test_split(raw_df,
                                   test_size=config.split_data.test_size,
                                   random_state=config.base.random_state)
    
    train.to_csv(Path.cwd()/config.split_data.train_path, index=False, sep=',')
    test.to_csv(Path.cwd()/config.split_data.test_path, index=False, sep=',')
    
    logger.info('Saved train and test set !!!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    split_dataset()