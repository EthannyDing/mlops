# simple xgboost example
This repo shows a simple example of using MLflow to track model training, reproduce results and model versioning

## Installation

`$ pip install mlflow`

## Model Tracking

Run `train.py` or `hyparam_search.py` individually

$ cd simple_xgboost_example

$ python train.py --learning-rate=0.1 -P colsample-bytree=0.7

$ python hyparam_search.py


### 1. Configure your backend database store

You can create a backend database store either in local database or remote ones. Here we will create a local Postgres DB call "models" as backend store. 
You can either create a Postgres Database programmatically or in the Postico UI. Hence DB URI:
`postgresql://postgres:knowtions12345@localhost:5432/models`


To be continued
