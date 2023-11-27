"""
date_time             0.333333
number                0.333333
integer               0.166667
bunho                 0.666667
email                 0.166667
url                   0.166667
part_num              0.666667
part_text             0.333333
part_discriminator         1.0
part_mask                  0.0
part_minus                 0.0
len_purity            0.194444
value_nunique               12
value_distr           3.584963
datatype                object
BUNHO                        0
NALJJA                       0
MYEONG                       0
JUSO                         0
YEOBU                        0
CODE                         0
ID                           0
SURYANG                   True
GEUMAEK                      0
NAEYOUNG                     0
YUL                          0
ETC                          0
"""

labels = ['BUNHO', 
          'NALJJA', 
          'MYEONG', 
          'JUSO', 
          'YEOBU', 
          'CODE', 
          'ID', 
          'SURYANG', 
          'GEUMAEK', 
          'NAEYOUNG', 
          'YUL'
]

features = ['date_time', 'number', 'integer', 'bunho', 'email', 'url', 'part_num',
       'part_text', 'part_discriminator', 'part_mask', 'part_minus',
       'len_purity', 'value_nunique', 'value_distr', 'datatype', 'BUNHO',
       'NALJJA', 'MYEONG', 'JUSO', 'YEOBU', 'CODE', 'ID', 'SURYANG', 'GEUMAEK',
       'NAEYOUNG', 'YUL', 'ETC']

# 번호에 대한 전형적인 feature value 구성

import openpyxl
import pandas as pd

import sys
import random

from joblib import Parallel, delayed
import dask.dataframe as dd


import warnings
warnings.filterwarnings('ignore')
sys.path.append('/home/dwna/projects/domain_class')
from src.features.build_features import BuildFeatures




file_path = "quality_evaluation_data.xlsx"

df_e = pd.read_excel(file_path, sheet_name='error_samples')
df_n = pd.read_excel(file_path, sheet_name='normal_samples')

err_columns = ['table', 'col_nm', 'domain', 'rule', 'total_cnt', 'err_cnt', 'err_rt', 'err_ext_sql',
       'err_data']
nor_columns = ['table', 'col_nm', 'datatype', 'rule', 'domain', 'verify_type', 'err_free_data', 'act_date',
       'normal_data', 'others']
df_e.columns = err_columns
df_n.columns = nor_columns



columns = ['table', 'col_nm', 'domain', 'normal_data', 'err_data', 'datatype', 'rule',  'total_cnt', 'err_cnt', 'err_rt']
df = df_n.merge(df_e, 'left', on=['table','col_nm'],suffixes=('','_y')) 

df = (df[columns]
        [~df.normal_data.isna()]
        .assign(normal_data = df.normal_data.astype(str).str.split(',').apply(lambda x:[] if x ==['nan'] else x), 
                err_data = df.err_data.astype(str).str.split(',').apply(lambda x:[] if x ==['nan'] else x)
               )
)




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
                .compute(scheduler='processes')

    return result


# profiles = profile_data(df, abnormal_rate=0.01, length=1000)

# df_profiled = profile_data_dask(df, abnormal_rate=0.01, length=1000, number_of_partitions=5)

# profiles_job = profile_data_joblib(df, abnormal_rate=0.01, length=1000)