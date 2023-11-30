import os
import sys
import click
import json
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())

package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
sys.path.append(package_path)


import pandas as pd
from openpyxl import load_workbook
import warnings
# warnings.filterwarnings('ignore')

from src.features.build_features import BuildFeatures

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient




def load_best_model(experiment_name: str, metric: str = "metrics.Test_Accuracy"):
    """
    주어진 실험 이름과 메트릭을 사용하여 MLflow에서 최적의 모델을 로드합니다.

    :param experiment_name: 실험 이름.
    :param metric: 최적 모델을 선택하기 위한 메트릭 이름.
    :return: 최적의 모델 또는 실험이 없거나 실행을 찾지 못한 경우 None.
    """
    mlflow.set_tracking_uri(f"file://{package_path}/models")
    mlflow.set_experiment(experiment_name)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id, order_by=[f"{metric} DESC"])
    
        if runs:
            best_run = runs[0]
            best_run_id = best_run.info.run_id
            
            # 메트릭 이름에서 "metrics." 제거
            metric_name = metric.split('.')[-1]
            best_metric_value = best_run.data.metrics.get(metric_name)

            if best_metric_value is not None:
                print(f"Best run ID: {best_run_id} with {metric_name}: {best_metric_value}")

                model_uri = f"runs:/{best_run_id}/model"
                return mlflow.pyfunc.load_model(model_uri)
            else:
                print(f"Metric '{metric_name}' not found in the best run.")
        else:
            print("No runs found for the given search criteria.")
    else:
        print("Experiment not found.")

    return None

def estimate_domain(model, series: pd.Series, col_name: str = None):
    """
    주어진 series로부터 특징을 추출하고, 이를 사용하여 모델을 통해 예측값을 계산합니다.
    
    :param model: 예측을 수행할 모델 객체.
    :param series: 특징을 추출할 pd.Series, 리스트 또는 튜플.
    :param col_name: 컬럼 이름 (옵션).
    :return: 모델에 의해 예측된 결과.
    """
    
    if isinstance(series, (list, tuple)):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series, list, or tuple")

    # BuildFeatures를 사용하여 profiling patterns 실행
    profile = BuildFeatures(series, col_name).profiling_patterns()
    
    # 'col_name', 'datatype', 'domain' 인덱스 제거
    features = profile.drop(index=['col_name', 'datatype', 'domain'])

    # 모델의 predict 메서드 유효성 확인
    if not hasattr(model, 'predict'):
        raise AttributeError("Provided model does not have a predict method")

    return model.predict(features)

