MLOps Demo
==============================

This repository contains demo code to showcase the complete pipeline for 
using MLOps

Project Organization
------------

This Project is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

The demo code requires the following to work:
- AWS access key and secret key stored in the `.aws/credentials` file under the default namespace
- The AWS user should have access to Elastic Beanstalk, S3 and EC2 instance running MLFlow in it
- The raw dataset can be found in the [UCI website](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Upload the dataset to S3 and configure the bucket name in the `params.yaml` file

## AWS

- In the AWS configure a EC2 machine with port `8080` open and install `mlflow` and instantiate the server
- Update the `params.yaml` with the URL to the EC2 machine this will help to set the tracking uri for MLFlow

## Local Machine
- After updating the `params.yaml` file with appropriate paths run the `dvc repro` command from the project root
- The `result app` will be deployed to your AWS Elastic Beanstalk environment