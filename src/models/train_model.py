from data.transform_data import read_params
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
sys.path.append(str(Path.cwd()/'src/'))
sys.path.append(str(Path.cwd()/'src/data'))


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

#####################################################
    scores_file = str(Path.cwd()/config.reports.scores)
    params_file = str(Path.cwd()/config.reports.params)

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)
#####################################################

    logger.info('Model Saving !!!!')
    os.makedirs(model_dir, exist_ok=True)
    model_path = str(Path(model_dir/"model.joblib"))

    joblib.dump(lr, model_path)
    logger.info('Training and Evaluating finished Model Saved !!!!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_and_evaluate()
