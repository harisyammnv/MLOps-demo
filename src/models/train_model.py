import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import click
import logging
import json
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
sys.path.append(str(Path.cwd()/'src/'))
sys.path.append(str(Path.cwd()/'src/data'))
from data.transform_data import read_params

def eval_metrics(actual,  pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


@click.command()
@click.option('--config_path', help="Path to Configuration File")
def train_and_evaluate(config_path: str):

    logger = logging.getLogger()
    logger.info('Parsing the config file supplied')
    config = read_params(config_path)

    target_col = [config.base.target_col]
    alpha = config.estimators.ElasticNet.params.alpha
    l1_ratio = config.estimators.ElasticNet.params.l1_ratio
    model_dir = Path.cwd()/config.model_dir

    logger.info('Reading the Training and Test Data')
    train_df = pd.read_csv(config.split_data['train_path'], sep=',')
    test_df = pd.read_csv(config.split_data['test_path'], sep=',')

    train_y = train_df[target_col]
    test_y = test_df[target_col]

    train_x = train_df.drop(target_col, axis=1)
    test_x = test_df.drop(target_col, axis=1)
    
    mlflow_config = config.mlflow_config
    remote_tracking_uri = mlflow_config.remote_server_uri
    mlflow.set_tracking_uri(remote_tracking_uri)
    
    mlflow.set_experiment(mlflow_config.experiment_name)
    
    with mlflow.start_run(run_name=mlflow_config.run_name)as mlops_run:
        logger.info('Training Started Data')
        lr = ElasticNet(alpha=alpha,
                        l1_ratio=l1_ratio,
                        random_state=config.base.random_state)

        lr.fit(train_x, train_y)
        logger.info('Training finished')

        logger.info('Predicting on the Test Data')
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        logger.info('Model Saving !!!!')
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name=mlflow_config.registered_model_name)
        else:
            mlflow.sklearn.log_model(lr, "model")
        
        logger.info('Training and Evaluating finished Model Saved !!!!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_and_evaluate()
