# -*- coding: utf-8 -*-
import logging
import pandas as pd
import click
import yaml
from dotmap import DotMap
from fetch_data import S3DataFetch
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


class TransfromData:

    def __init__(self, config: DotMap, filename: str):

        self.config = config
        self.input_path = self.config.load_data['raw_dataset_path']
        self.output_path = self.input_path
        self.filename = filename
    
    def tranform_raw_data(self):
        df = pd.read_csv(Path.cwd()/self.input_path/self.filename, sep=';')
        ### Include more transformations possible
        df.to_csv(Path.cwd()/self.output_path/self.filename, sep=',', index=False)


def read_params(config_path: str):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return DotMap(config)


@click.command()
@click.option('--config_path', help="Path to Configuration File")
def make_dataset(config_path: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger()
    logger.info('Parsing the config file supplied')

    config = read_params(config_path)
    
    logger.info('Downloading data from S3')
    data_fetch = S3DataFetch(bucket_name=config.data_source['s3_bucket'],
                             file_name=config.data_source['filename'],
                             save_path=Path.cwd()/config.load_data['raw_dataset_path'])
    data_fetch.get_file()
    logger.info('Saved the file to local directory')
    
    logger.info('Initial Transofrmation started...')
    
    td = TransfromData(config=config, filename=config.data_source['filename'])
    td.tranform_raw_data()
    
    logger.info('Initial Transofrmation finished!!!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    make_dataset()