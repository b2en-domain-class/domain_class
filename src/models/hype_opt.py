import os
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
import json
import click

# Load the .env file
load_dotenv(find_dotenv())

package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
sys.path.append(package_path)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval


# Extend the model_spaces dictionary with hyperparameter spaces for SVM, LightGBM, and CatBoost
model_spaces = {
    'logistic_regression': {
        'preprocessing': hp.choice('lr_preprocessing', ['standard', 'minmax']),
        'C': hp.loguniform('lr_C', -4, 4)
    },
    'random_forest': {
        'preprocessing': hp.choice('rf_preprocessing', ['standard', 'minmax']),
        'n_estimators': hp.choice('rf_n_estimators', [10, 50, 100, 200]),
        'max_depth': hp.choice('rf_max_depth', [5, 10, 20, None])
    },
    'svm': {
        'preprocessing': hp.choice('svm_preprocessing', ['standard', 'minmax']),
        'C': hp.loguniform('svm_C', -4, 4),
        'gamma': hp.loguniform('svm_gamma', -4, 4)
    },
    'lgbm': {
        'preprocessing': hp.choice('lgbm_preprocessing', ['standard', 'minmax']),
        'learning_rate': hp.loguniform('lgbm_learning_rate', -4, 0),
        'n_estimators': hp.choice('lgbm_n_estimators', [10, 50, 100, 200]),
        'num_leaves': hp.choice('lgbm_num_leaves', [15, 31, 63, 127]), 
        'max_depth': hp.choice('lgbm_max_depth', [5, 10, 20, -1])
    },
    'catboost': {
        'preprocessing': hp.choice('catboost_preprocessing', ['standard', 'minmax']),
        'learning_rate': hp.loguniform('catboost_learning_rate', -4, 0),
        'iterations': hp.choice('catboost_iterations', [10, 50, 100, 200]),
        'depth': hp.choice('catboost_depth', [4, 6, 8, 10])
    }
}

# Update the create_model function
def create_model(model_name, params):
    classifier_dic = { 
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'lgbm': LGBMClassifier,
            'catboost': CatBoostClassifier,
    }
    return classifier_dic[model_name](**params)

# Function to choose preprocessing based on the parameter
# def choose_preprocessing(preprocessing_choice):
#     if preprocessing_choice == 'standard':
#         return StandardScaler()
#     elif preprocessing_choice == 'minmax':
#         return MinMaxScaler()


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
    

@click.command()
@click.option('--model_name', prompt='Enter the model name (e.g., logistic_regression, random_forest, svm, lgbm, catboost)',
              help='Name of the model.')
@click.option('--train_data_path', prompt='Now enter the training data path',
              default='/data/processed/profiles/2/ver_2_len_1000_rate_0.01.csv',
              help='training data relative path from package path',
              show_default=True)
def main(model_name:str, train_data_path:str):
    # Load data
    data = load_data(train_data_path)
    data = data.dropna()
    # Split data into features and target
    X = data.drop(columns=['col_name', 'datatype', 'domain' ])
    y = data['domain']
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
    
    # Objective function for optimization
    def objective(model_params, model_name):
    
        # Create a column transformer for preprocessing
        preprocessing_choice = model_params.pop('preprocessing') if 'preprocessing' in model_params else 'standard'
        preprocessing = choose_preprocessing(preprocessing_choice)
        # Create and train the pipeline with the best model
        model = create_model(model_name, model_params)   
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', model)
        ])
    
        score = cross_val_score(pipeline, X, y, cv=3, n_jobs=-1).mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=lambda params: objective(params, model_name),
                space=model_spaces[model_name],
                algo=tpe.suggest,
                max_evals=50,  # Adjust as needed
                trials=trials)
    best_params = space_eval(model_spaces[model_name], best)
    print(f"Best parameters for {model_name}: {json.dumps(best_params)}")
    return {model_name: best_params}

if __name__ == "__main__":
    main()
      
      
# Best parameters for logistic_regression: {"C": 35.97694901277438, "preprocessing": "standard"}
# Best parameters for random_forest: {"max_depth": 20, "n_estimators": 100, "preprocessing": "standard"}
# Best parameters for svm: {"C": 3.4646160593658917, "gamma": 1.45111800508685, "preprocessing": "standard"}
# Best parameters for lgbm: {"learning_rate": 0.0937282043841488, "max_depth": 10, "n_estimators": 50, "num_leaves": 15, "preprocessing": "standard"}
# Best parameters for catboost: {"depth": 10, "iterations": 100, "learning_rate": 0.29681845674147794, "preprocessing": "minmax"}


# Best parameters for lgbm: {"learning_rate": 0.17528711814661266, "max_depth": 5, "n_estimators": 50, "num_leaves": 127, "preprocessing": "minmax"}
# Best parameters for catboost: {"depth": 10, "iterations": 100, "learning_rate": 0.233387197467268, "preprocessing": "standard"}
# Best parameters for svm: {"C": 8.864809889251712, "gamma": 1.783460164495912, "preprocessing": "standard"}
# Best parameters for logistic_regression: {"C": 50.55202347085015, "preprocessing": "minmax"}
# Best parameters for random_forest: {"max_depth": 20, "n_estimators": 200, "preprocessing": "standard"}