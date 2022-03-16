# mlops
This repo shows a simple example of using MLflow to track model training, reproduce results and model versioning

## Installation
    $ pip install mlflow

## Model Tracking

Run `train.py` or `hyparam_search.py` individually

    $ cd simple_xgboost_example

    $ python train.py --learning-rate=0.1 -P colsample-bytree=0.7

    $ python hyparam_search.py

## Running MLflow in Docker

The server will run as a stack of three containers: mlflow_server, postgresdb and nginx. Any experiments and registered models are backed up in 

- AWS S3: for model artifacts, e.g. model binary file, training data and other large data files.

- Postgres DB: for storing model metrics, parameters, tags, descriptions and other metadata.
### 1. Environment Variables
- POSTGRES_USER: postgres db user name 
- POSTGRES_PASSWORD: postgres db password
- POSTGRES_HOST: postgres db host name, e.g. localhost or remote db server IP address 
- POSTGRES_DB: initial database to be created and in which MLflow tables will be created
- AWS_ACCESS_KEY_ID: AWS_ACCESS_KEY_ID if storing artifacts in S3
- AWS_SECRET_ACCESS_KEY: AWS_SECRET_ACCESS_KEY if storing artifacts in S3
- AWS_DEFAULT_REGION: AWS_DEFAULT_REGION if storing artifacts in S3
- AWS_S3_MLFLOW_ARTIFACTS: AWS S3 rootpath

For security purpose, you may save these environment variables in your `.env` file or `export` them with bash in terminal.

### 2. Start MLFlow Tracking Server

You can start the tracking server by executing the following commands

    $ cd docker
    $ docker-compose --env-file <.env filepath> -d up

