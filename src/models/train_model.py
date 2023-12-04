import os
import sys
import click
import json
import tempfile
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

# Initialize MLflow tracking
mlflow.set_tracking_uri(f"http://localhost:5000")  
mlflow.set_experiment("hyperParemeterOpted")  # Replace with your experiment name

def load_data(file_path):
    file_path = package_path + file_path
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"파일이 비어있습니다: {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"파일을 파싱하는 데 실패했습니다: {file_path}")
        return None
    except Exception as e:
        print(f"데이터를 불러오는데 실패했습니다: {e}")
        return None

# Update the create_model function
def create_model(model_name, params):
    if model_name == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_name == 'svm':
        model = SVC(**params)
    elif model_name == 'lgbm':
        model = LGBMClassifier(**params)
    elif model_name == 'catboost':
        model = CatBoostClassifier(**params, verbose=0)
    return model


def show_params(ctx, param, value):
    model_spaces = {
        "logistic_regression": {
            "preprocessing": ["standard", "minmax"],
            "C": (-4, 4)
        },
        "random_forest": {
            "preprocessing": ["standard", "minmax"],
            "n_estimators": [10, 50, 100, 200],
            "max_depth":[5, 10, 20, None]
        },
        "svm": {
            "preprocessing": ["standard", "minmax"],
            "C": (-4, 4),
            "gamma": (-4, 4)
        },
        "lgbm": {
            "preprocessing": ["standard", "minmax"],
            "learning_rate":  "10^(-4, 0)",
            "n_estimators":  [10, 50, 100, 200],
            "num_leaves": [15, 31, 63, 127], 
            "max_depth":  [5, 10, 20, -1]
        },
        "catboost": {
            "preprocessing": ["standard", "minmax"],
            "learning_rate": "10^(-4, 0)",
            "iterations": [10, 50, 100, 200],
            "depth": [4, 6, 8, 10]
        }
    }
    
    if ctx.obj is None:
        ctx.obj = {}

    if ctx.obj.get('model_name_shown'):
        return value

    if value in model_spaces:
        print(f"Parameter space for {value}: {json.dumps(model_spaces[value])}")
        ctx.obj['model_name_shown'] = True
    else:
        raise click.BadParameter(f"Invalid model name: {value}. Available models: {list(model_spaces.keys())}")
    return value

def log_dict_as_artifact(results_dict, artifact_path='model_metrics'):
    """
    Logs a dictionary as a JSON artifact named 'report.json' to the current MLflow run, then deletes the file.

    Parameters:
    - results_dict (dict): The dictionary to log.
    - artifact_path (str): The artifact directory in the MLflow run to store the artifact.
    """
    # Define the report file name
    report_filename = 'metrics_report.json'
    
    # Serialize and save the dictionary as JSON
    with open(report_filename, 'w') as report_file:
        json.dump(results_dict, report_file, indent=4)
    
    # Log the report file as an artifact
    mlflow.log_artifact(report_filename, artifact_path)
    
    # Delete the report file
    os.remove(report_filename)


@click.command()
@click.option('--model_name', prompt='Enter the model name (e.g., logistic_regression, random_forest, svm, lgbm, catboost)',
              help='Name of the model.', callback=show_params)
@click.option('--params', prompt='Now enter the model parameters in JSON format (e.g., {"C": 1.0})',
              default={},
              help='JSON string of model parameters.',
              show_default=True)
@click.option('--train_data_path', prompt='Now enter the training data path )',
              default='/data/processed/profiles/2/ver_2_len_1000_rate_0.01.csv',
              help='training data relative path from package path',
              show_default=True)
def main(model_name:str, params:str, train_data_path:str):
    """
    주어진 모델 이름과 JSON 형식의 모델 파라미터를, 프로파일 트레이닝데이터 패스를 입력으로 하여
    모델을 학습하고 로그 및 모델을 mlflow frame에 저장합니다.

    Args:
    model_name (str): 모델의 이름.
    model_params (str): JSON 형식의 문자열로 표현된 모델 파라미터.
    train_data_path (str): 학습할 데이터 셋의 상대경로. 학습데이터 루트경로는 package_path/data/processed/profiles
    """
    try:
        model_params = json.loads(params)
    except json.JSONDecodeError:
        raise ValueError("params must be a valid JSON string.")
    
    # Load train data
    data = load_data(train_data_path)
    data = data.dropna()
    # Split data into features and target
    X = data.drop(columns=['col_name', 'datatype', 'domain' ])
    y = data['domain']
    
    # Load test data
    parts = train_data_path.split('/')
    parts.insert(5, 'test')
    test_data_path = '/'.join(parts)
    test_data = load_data(test_data_path)
    test_data = test_data.dropna()
    # Split data into features and target
    X_test = test_data.drop(columns=['col_name', 'datatype', 'domain' ])
    y_test = test_data['domain']
        

    # Define continuous and binary columns
    # continuous column names start with lowercase letter. 
    # category columns start with uppercase letter 
    continuous_cols = [col for col in X.columns if col[0].islower()]
    binary_cols = [col for col in X.columns if col[0].isupper()]
    
    # Function to choose preprocessing based on the parameter
    def choose_preprocessing(preprocessing_choice):
        if preprocessing_choice == 'minmax':
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), continuous_cols),
                    # ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), binary_cols)
                ],
                remainder='passthrough'
            )
        else: 
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), continuous_cols),
                    # ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), binary_cols)
                ],
                remainder='passthrough'
            )
        return preprocessor
    
    # Create a column transformer for preprocessing
    preprocessing_choice = model_params.pop('preprocessing') if 'preprocessing' in model_params else 'standard'
    preprocessing = choose_preprocessing(preprocessing_choice)
    # Create and train the pipeline with the best model
    model = create_model(model_name, model_params)   
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', model)
    ])

    # MLflow tracking
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)
        
        # Train the model on the training set
        pipeline.fit(X, y)
    
        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)
    
        # Calculate the accuracy on the test set
        accuracy = accuracy_score(y_test, y_pred)
        print(model_params)
        print(f'{model_name} Test Accuracy: {accuracy:.2f}')
        mlflow.log_metric("Test_Accuracy", accuracy)
        
        class_report = classification_report(y_test, y_pred, output_dict=True)   
        log_dict_as_artifact(class_report, "model_metrics")
        
        
        # # Cross-validation
        # cv_scores = cross_val_score(pipeline, X, y, cv=5)
        # print(f'{model_name} accuracy: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}')

        # # Log metrics
        # mlflow.log_metric("cv_accuracy_mean", np.mean(cv_scores))
        # mlflow.log_metric("cv_accuracy_std", np.std(cv_scores))

        # Log the pipeline
        mlflow.sklearn.log_model(pipeline, "model") 
        mlflow.set_tag("model", model_name)
        
if __name__ == "__main__":
    main()