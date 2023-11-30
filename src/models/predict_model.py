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



# Set your MLflow tracking URI and experiment name
mlflow.set_tracking_uri(f"file://{package_path}/models")
mlflow.set_experiment("hyperParemeterOpted")

client = MlflowClient()

# Get the experiment ID using the experiment name
experiment = client.get_experiment_by_name("hyperParemeterOpted")
if experiment:
    experiment_id = experiment.experiment_id

    # Define a search filter using the SQL-like syntax
    filter_string = "metrics.rmse < 1.0"  # Modify this to your metric condition

    # Search the runs in the specified experiment and order by the metric descending
    runs = client.search_runs(experiment_id, filter_string, order_by=["metrics.rmse DESC"])

    # Assuming you want the run with the highest metric value
    if runs:
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_metric_value = best_run.data.metrics['rmse']

        print(f"Best run ID: {best_run_id} with RMSE: {best_metric_value}")

        # Construct the model URI
        model_uri = f"runs:/{best_run_id}/model"
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        # Now you can use the model for inference or further evaluation
    else:
        print("No runs found for the given search criteria.")
else:
    print("Experiment not found.")