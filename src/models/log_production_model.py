import sys
import joblib
import mlflow
import logging
import click
from pprint import pprint
from pathlib import Path
sys.path.append(str(Path.cwd()/'src/'))
sys.path.append(str(Path.cwd()/'src/data'))
from data.transform_data import read_params
from mlflow.tracking import MlflowClient


@click.command()
@click.option('--config_path', help="Path to Configuration File")
def log_production_model(config_path):
    
    logger = logging.getLogger()
    logger.info('Parsing the config file supplied')
    config = read_params(config_path)
    
    
    mlflow_config = config.mlflow_config
    

    model_name = mlflow_config.registered_model_name


    remote_server_uri = mlflow_config.remote_server_uri

    mlflow.set_tracking_uri(remote_server_uri)
    logger.info('Setting the Tracking URI')
    
    runs = mlflow.search_runs(experiment_ids=1)
    lowest = runs["metrics.mae"].sort_values(ascending=True).min()
    lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"].iloc[0]
    
    logger.info('Obtaining the model with the best metric')
    
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )


    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_dir = Path.cwd()/config.model_dir
    model_path = str(Path(model_dir/"model.joblib"))

    joblib.dump(loaded_model, model_path)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    log_production_model()