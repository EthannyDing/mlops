export MLFLOW_TRACKING_URI="postgresql://postgres:knowtions12345@localhost:5432/models"
mlflow models serve -m "models:/SimpleXgboostModel/1" --no-conda
