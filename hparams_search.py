import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl


import mlflow
import mlflow.xgboost

mpl.use("Agg")


# def parse_args():
#     parser = argparse.ArgumentParser(description="XGBoost example")
#     parser.add_argument(
#         "--learning-rate",
#         type=float,
#         default=0.3,
#         help="learning rate to update step size at each boosting step (default: 0.3)",
#     )
#     parser.add_argument(
#         "--colsample-bytree",
#         type=float,
#         default=1.0,
#         help="subsample ratio of columns when constructing each tree (default: 1.0)",
#     )
#     parser.add_argument(
#         "--subsample",
#         type=float,
#         default=1.0,
#         help="subsample ratio of the training instances (default: 1.0)",
#     )
#     return parser.parse_args()


def mlflow_run(param, dtrain, dtest, y_test):

    with mlflow.start_run() as run:

        run_id = run.info.run_id
        # train model
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": param["learning_rate"],
            "eval_metric": "mlogloss",
            "colsample_bytree": param["colsample_bytree"],
            "subsample": 1.0,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

    return run_id, loss, acc


def main():
    # parse command-line arguments
    # args = parse_args()
    params = [{"learning_rate": 0.1, "colsample_bytree": 0.5},
              {"learning_rate": 0.2, "colsample_bytree": 0.5},
              {"learning_rate": 0.3, "colsample_bytree": 0.5},
              {"learning_rate": 0.1, "colsample_bytree": 0.7},
              {"learning_rate": 0.1, "colsample_bytree": 0.9},
              ]

    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # set tracking uri as local DB.
    # mlflow.set_tracking_uri("postgresql://postgres:knowtions12345@localhost:5432/models")

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    for param in params:
        run_id, loss, acc = mlflow_run(param, dtrain, dtest, y_test)
        print("Run ID: {}\nLog-loss: {}, Accuracy: {}".format(run_id, loss, acc))


if __name__ == "__main__":
    main()
