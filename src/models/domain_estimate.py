import os
import sys
import click
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())

package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
sys.path.append(package_path)


import pandas as pd
from joblib import Parallel, delayed

import warnings
# warnings.filterwarnings('ignore')


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
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


from src.features.build_features import BuildFeatures
from src.util.files import load_csv, write_csv


# Initialize MLflow tracking
mlflow.set_tracking_uri(f"http://localhost:5000")  


class ModelLoader:
    def __init__(self, model_name=None, experiment_name=None, stage='Production'):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.model = None
        self.stage = stage

    def load_model(self):
        """MLflow에서 모델을 불러옵니다. model_name 또는 experiment_name을 기반으로 합니다."""
        if self.model_name:
            model_uri = f"models:/{self.model_name}/{self.stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
        elif self.experiment_name:
            # experiment_name을 사용하여 모델 로드하는 로직 구현
            # 예를 들어, experiment_id를 찾고, 해당 experiment의 최신 run에서 모델을 로드할 수 있습니다.
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs([experiment.experiment_id])
                latest_run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{latest_run_id}/model"
                self.model = mlflow.pyfunc.load_model(model_uri)
            else:
                raise ValueError(f"No experiment found with name '{self.experiment_name}'")
        else:
            raise ValueError("Model name or experiment name must be provided")

    def predict(self, data):
        """데이터에 대한 예측을 수행합니다."""
        if self.model is not None:
            return self.model.predict(data)
        else:
            raise Exception("Model is not loaded. Call load_model() first.")



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
                # print(f"Best run ID: {best_run_id} with {metric_name}: {best_metric_value}")

                model_uri = f"runs:/{best_run_id}/model"
                return mlflow.pyfunc.load_model(model_uri)
            else:
                print(f"Metric '{metric_name}' not found in the best run.")
        else:
            print("No runs found for the given search criteria.")
    else:
        print("Experiment not found.")

    return None


def prepare_row(row):
    warnings.filterwarnings('ignore')
    col_values = eval(row.col_values) if isinstance(row.col_values, str) else row.col_values
    series = pd.Series(col_values, name=row.col_nm)
    profile = BuildFeatures(series, domain=row.domain).profiling_patterns()
    return profile

def prepare_data_joblib(df: pd.DataFrame, num_cores:int=-1) -> pd.DataFrame:
    # Prepare the data for parallel processing
    results = Parallel(n_jobs=num_cores)(delayed(prepare_row)(row) 
                                  for row in df.itertuples(index=False))
    return pd.DataFrame(results)


def estimate_domain(model, data_file_path:str):
    """
    주어진 series로부터 특징을 추출하고, 이를 사용하여 모델을 통해 예측값을 계산합니다.
    
    :param model: 예측을 수행할 모델 객체.
    :param series: 특징을 추출할 pd.Series, 리스트 또는 튜플.
    :param col_name: 컬럼 이름 (옵션).
    :return: 모델에 의해 예측된 결과.
    """
    
    # load data
    # data_file_path = '/data/processed/table/pqt/0.parquet'
    data_file_path = package_path + data_file_path
    
    # 입력 파일 확장자에 따라 적절한 읽기 방법 선택
    file_extension = os.path.splitext(data_file_path)[-1]
    if file_extension == '.parquet':
        df = pd.read_parquet(os.path.join(package_path, data_file_path))
    elif file_extension == '.csv':
        df = pd.read_csv(os.path.join(package_path, data_file_path))
    else:
        raise ValueError("Unsupported file format")
    
        
    # BuildFeatures를 사용하여 profiling patterns 실행
    profiles = prepare_data_joblib(df)
    
    # 'col_name', 'datatype', 'domain' 컬럼 제거
    cols_to_drop = ['col_name', 'datatype', 'domain']
    cols_to_drop = [col for col in cols_to_drop if col in profiles.columns]
    
    features = profiles.drop(columns=cols_to_drop) if cols_to_drop else profiles

    # 모델의 predict 메서드 유효성 확인
    if not hasattr(model, 'predict'):
        raise AttributeError("Provided model does not have a predict method")
    result = model.predict(features)
    result = pd.DataFrame(result, columns=['domain'])
    
    return result






def prompt_model_or_experiment():
    if click.confirm('Do you want to load a registered model?', default=True):
        return click.prompt('Enter the registered model name', default=None), None
    else:
        return None, click.prompt('Enter the experiment name', default=None)

@click.command()
@click.option('--model_name', default=None, hidden=True)
@click.option('--experiment_name', default=None, hidden=True)
@click.option('--stage', prompt='If registered model, enter the stage (Staging, Production, Archived)', default='Production', help='Model stage for loading. Default is "Production".')
@click.option('--data_path', prompt='Now enter the parquet filepath', default='/data/processed/table/pqt/0.parquet', help='Path to the data file for domain estimation.')
def main(model_name, experiment_name, stage, data_path):
    
    if not model_name and not experiment_name:
        model_name, experiment_name = prompt_model_or_experiment()

    # model = load_best_model(experiment_nm)
    model = ModelLoader(model_name, experiment_name, stage)
    model.load_model()
    result = estimate_domain(model, data_path)
    
    data_path = package_path + data_path
    directory, filename = os.path.split(data_path)
    file_basename, file_extension = os.path.splitext(filename)
    output_file = os.path.join(directory, f'{file_basename}_result.csv')
    result.to_csv(output_file, index=False)
    # write_csv(result,  'result_of_domain_estimation_'+ data_path[:-3] )
    print(result.head())
    return None




if __name__ == '__main__':
    main()
        
        
        
        
        
    # # 가상 컬럼 데이터 생성
    # from src.data.make_trainingdataset import generate_combined_set
    # data = []
    # for col in ['aaa', 'bbb', 'ccc']:
    #     col_data = generate_combined_set(normal_list=['abc','efaa','ddga'], 
    #                                      abnormal_list=['abc','1ab'], 
    #                                      abnormal_probability=0.001, 
    #                                      length=1000)
    #     col_name = col
    #     series = pd.Series(col_data, name = col_name)
    #     data.append(series)
    # df = pd.concat(data, axis=1)
    # df.to_csv('test.csv', index=False)
    # # 모델 로드
    # model = load_best_model('hyperParemeterOpted')
    # # 컬럼 도메인 추정
    # print(estimate_domain(model, 'test.csv'))
    
    
#     # 모델 로더 객체 생성 및 모델 불러오기
#     loader = ModelLoader(model_name)
#     loader.load_model()

#     # 예측을 위한 샘플 데이터
#     # 실제 사용시에는 모델에 맞는 데이터 형식을 사용해야 합니다.
#     sample_data = [...]  # 샘플 데이터 입력

#     # 예측 수행
#     predictions = loader.predict(sample_data)
#     print(predictions)
