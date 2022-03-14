from ctypes.wintypes import tagMSG
from pathlib import Path
import xgboost as xgb
import matplotlib as mpl
import pprint

import mlflow
import mlflow.xgboost

hq_postgres_uri = 'postgresql://knowtions:verystrongwarehousepass@192.168.10.11:5432/models'
local_postgres_uri = 'postgresql://postgres:ethanding12345@localhost:5432/models'
s3_bucket = "s3://mlops-buckets/mlartifacts"


def get_cancer_v1():

    rootpath = Path('/Users/ethanding/knowtions/eucleia')
    artifacts =  {
        'features_path': rootpath / 'python/eucleia/bin/twl/models/v1/cancer_features.pkl',
        'model_path': rootpath / 'python/eucleia/bin/twl/models/v1/cancer_model.json',
        }
    artifacts['model'] = xgb.Booster(model_file=artifacts['model_path'])
    return artifacts

def get_cancer_v2():

    rootpath = Path('/Users/ethanding/knowtions/eucleia/python/eucleia/bin/twl')
    artifacts =  {
        'features_path': rootpath / 'models/v2/cancer_features_version2.json',
        'model_path': rootpath / 'models/v2/cancer_version2.model',
        }
    artifacts['model'] = xgb.Booster(model_file=artifacts['model_path'])
    return artifacts

def delete_registered_model(model_name):
    client = mlflow.tracking.MlflowClient(tracking_uri=hq_postgres_uri)
    client.delete_registered_model(model_name)


def collect_models(model_names, age_bins, version):

    rootpath = Path('/Users/ethanding/knowtions/eucleia/python/eucleia/bin/twl')
    
    artifacts = {name: {} for name in model_names}
    for model_name in model_names:
        model_rootpath = rootpath / f'models/{version}'
        model_files = list(model_rootpath.iterdir())
        
        artifacts[model_name]['feature_path'] = [f for f in model_files if f'{model_name}_features' in f.__str__()][0]
        artifacts[model_name]['model_path'] = [f for f in model_files if (model_name in f.__str__()) and (f.__str__().endswith('.model'))][0]

        for age_bin in age_bins:
            transformer_rootpath = rootpath / f'transformers/{version}'
            transformer_files = list(transformer_rootpath.iterdir())
            artifacts[model_name]['transformer_' + age_bin] = [f for f in transformer_files if f'{age_bin}_{model_name}' in f.__str__()][0]

    pprint.pprint(artifacts)
    return artifacts


def collect_v1_models():
    model_names = [
        'cancer',
        'dementia', 
        'hospitalization', 
        'mortality', 
        'cardio_cerebro_vascular_diseases',
        'chronic_kidney_disease',
        'chronic_liver_disease',
        'COPD',
        'diabetes'
        ]
    age_bins = ['10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80+']
    
    return collect_models(model_names, age_bins, version='v1')


def collect_v2_models():
    model_names = [
        'cancer',
        'dementia', 
        'hospitalization', 
        'mortality', 
        ]
    age_bins = ['10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90', '90+']
    
    return collect_models(model_names, age_bins, version='v2')


def collect_v3_models():
    model_names = [
        'major_diseases',
        'hospitalization',
        'cardio_cerebro_vascular_diseases',
        'chronic_kidney_disease',
        'chronic_liver_disease',
        'copd',
        'diabetes'
        ]
    age_bins = ['10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90', '90+']
    
    return collect_models(model_names, age_bins, version='v3')


def collect_v3_1_models():
    model_names = [
        'dementia',
        ]
    age_bins = ['10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90', '90+']
    
    return collect_models(model_names, age_bins, version='v3_1')


def main():
    expr_name = 'register v3_1 models re'
    mlflow.set_tracking_uri(hq_postgres_uri)
    mlflow.create_experiment(expr_name, s3_bucket)
    mlflow.set_experiment(expr_name)

    artifacts = collect_v3_1_models()
    for model_name in artifacts.keys():
        print("Registering {} in tracking server".format(model_name))
        with mlflow.start_run() as run:

            # mlflow.log_metrics({'accuracy': 0.56, 'mse': 2.1})
            # mlflow.log_params({'weight': 3, 'features_num': 30, 'lr': 0.001})
            for art in artifacts[model_name].values():
                mlflow.log_artifact(art.__str__())

            # register logged model artifacts
            print(run.info)

            model_uri = f"runs:/{run.info.run_id}/artifacts/{artifacts[model_name]['model_path'].name}"
            print(f'model uri: {model_uri}')

            mlflow.register_model(
                model_uri=model_uri, 
                name=model_name,
            )
            # mlflow.xgboost.log_model(
            #     xgb_model=artifacts['model'],
            #     artifact_path=s3_bucket + '/' + run.info.run_id + '/artifacts',
            #     registered_model_name="brand_new_cancer",
            # )


if __name__ == '__main__':

    main()
    # collect_v1_models()
    # collect_v2_models()
    # collect_v3_models()
    # collect_v3_1_models()