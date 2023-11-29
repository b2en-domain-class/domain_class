import sys, os
import warnings
from dotenv import load_dotenv, find_dotenv

import pandas as pd
import random
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import dask.dataframe as dd
import click

# Load the .env file
load_dotenv(find_dotenv())
package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
sys.path.append(package_path)
from src.features.build_features import BuildFeatures

def load_excel_data(file_path):
    try:
        data = pd.read_excel(file_path)
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



def generate_combined_set(normal_list, abnormal_list, abnormal_probability, length):
    # Early return if both lists are empty
    if not normal_list and not abnormal_list:
        return []

    # Adjust counts if one of the lists is empty
    abnormal_count = int(length * abnormal_probability) if abnormal_list else 0
    normal_count = length - abnormal_count

    # Generate lists based on the counts
    chosen_abnormal = random.choices(abnormal_list, k=abnormal_count) if abnormal_list else []
    chosen_normal = random.choices(normal_list, k=normal_count) if normal_list else []

    # Combine and shuffle the final list
    final_list = chosen_normal + chosen_abnormal
    random.shuffle(final_list)

    return final_list


def profile_data(df: pd.DataFrame, abnormal_rate: float, length: int)->pd.DataFrame:
    """
    Profile the data from a DataFrame using the BuildFeatures class.
    """
    profiles = []

    for row in df.itertuples(index=False):
        combined = generate_combined_set(row.normal_data, row.err_data, abnormal_rate, length)
        series = pd.Series(combined, name=row.col_nm)

        profile = BuildFeatures(series, domain=row.domain).profiling_patterns()
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def profile_row(row, abnormal_rate, length):
    warnings.filterwarnings('ignore')
    combined = generate_combined_set(row.normal_data, row.err_data, abnormal_rate, length)
    series = pd.Series(combined, name=row.col_nm)
    profile = BuildFeatures(series, domain=row.domain).profiling_patterns()
    return profile

def profile_data_joblib(df: pd.DataFrame, abnormal_rate: float, length: int, num_cores:int=-1) -> pd.DataFrame:
    # Prepare the data for parallel processing
    results = Parallel(n_jobs=num_cores)(delayed(profile_row)(row,abnormal_rate, length) 
                                  for row in df.itertuples(index=False))
    return pd.DataFrame(results)

def profile_data_dask(df: pd.DataFrame, abnormal_rate: float, length: int, number_of_partitions:int =10) -> pd.DataFrame:
    # Prepare the data for parallel processing
    ddf = dd.from_pandas(df, npartitions=number_of_partitions)
    result = ddf.map_partitions(lambda partition: partition.apply(lambda row: profile_row(row,  abnormal_rate, length), axis=1)) \
                .compute(scheduler='processes').dropna()

    return result

def to_csv_with_directory(df, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(file_path, index=False)


# profiles = profile_data(df, abnormal_rate=0.01, length=1000)
@click.command()
@click.option('--source_data_path', prompt='Now enter the source data path (e.g.,/source.xlsx)',
              default='/source.xlsx',
              help='source data relative path',
              show_default=True)
@click.option('--output_data_path', prompt='Now enter the output data path for training(e.g., /1/ver_1_len_1000_rate_0.01.csv)',
              default='/1/ver_1_len_1000_rate_0.01.csv',
              help='training data relative path',
              show_default=True)
def main(source_data_path:str, output_data_path:str ):
    df = load_excel_data(package_path + '/data/external' + source_data_path)
    profiles_job = profile_data_joblib(df, abnormal_rate=0.01, length=1000)
    train_df, test_df = train_test_split(profiles_job, test_size=0.1, random_state=42)
    profile_data_path = package_path + '/data/processed/profiles'
    to_csv_with_directory(train_df,  profile_data_path + output_data_path)

    parts = output_data_path.split('/')
    parts.insert(2, 'test')
    test_data_path = '/'.join(parts)
    to_csv_with_directory(test_df,  profile_data_path + test_data_path)
    
if __name__ == "__main__":
    main()