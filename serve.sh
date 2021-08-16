export MLFLOW_TRACKING_URI="postgresql://postgres:knowtions12345@localhost:5432/models"
mlflow models serve -m "models:/XGboostModel/1"
