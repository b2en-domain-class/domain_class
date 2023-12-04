import sys, os
import warnings
from dotenv import load_dotenv, find_dotenv

import pandas as pd
import random
import re, string

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
    file_path = package_path + file_path
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


def is_korean_char(ch):
    """Check if a character is a Korean character."""
    return '\uAC00' <= ch <= '\uD7A3'

def get_random_char(char_type):
    if char_type == 'digit':
        return str(random.randint(0, 9))
    elif char_type == 'lowercase':
        return random.choice(string.ascii_lowercase)
    elif char_type == 'uppercase':
        return random.choice(string.ascii_uppercase)
    elif char_type == 'korean':
        return chr(random.randint(0xAC00, 0xD7A3))
    
def find_pattern(s):
    keys = ["yn", "date_time", "number", "integer", "bunho", "email", "url"]
    patterns = {key: BuildFeatures.patterns[key] for key in keys}
    
    for key, pattern in patterns.items():
        if re.match(pattern, s, re.IGNORECASE):
            return pattern
    else:
        return None
        

def modify_string(s):
    """Modify a string by randomly replacing a character while ensuring it matches a regex pattern."""
    def get_char_type(ch):
        """Determine the character type."""
        if ch.isdigit():
            return 'digit'
        elif ch.isupper():
            return 'uppercase'
        elif ch.islower():
            return 'lowercase'
        elif is_korean_char(ch):
            return 'korean'
        else:
            return 'other'

    length = len(s)
    pattern = find_pattern(s)
    attempts = 0

    while attempts < 1000:  # Limit attempts to avoid infinite loops
        # Randomly pick an index
        index = random.randint(0, length - 1)

        char_type = get_char_type(s[index])
        # Skip if the character type is 'other'
        if char_type == 'other':
            attempts += 1
            continue

        # Replace the character at the selected index
        new_char = get_random_char(char_type)
        new_string = s[:index] + new_char + s[index + 1:]
        
        if pattern is None:
            return new_string

        # Check if the new string matches the pattern
        if re.match(pattern, new_string, re.IGNORECASE):
            return new_string

        attempts += 1

    return None  # Return None if no match is found within the attempt limit

# Example usage
# input_string = "Hello 123 안녕하세요"
# pattern = r"^[A-Za-z0-9\s]+안녕하세요$"  # Example pattern

# modified_string = modify_string(input_string, pattern)
# if modified_string:
#     print("Modified String:", modified_string)
# else:
#     print("No matching string found within the attempt limit.")


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
    
    if chosen_normal:
        chosen_normal = [modify_string(s) if s != 'NULL' else s 
                                        for s in chosen_normal if s]

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
        normal_list, abnormal_list = eval(row.normal_data), eval(row.err_data)
        combined = generate_combined_set(normal_list, abnormal_list, abnormal_rate, length)
        series = pd.Series(combined, name=row.col_nm)

        profile = BuildFeatures(series, domain=row.domain).profiling_patterns()
        profiles.append(profile)
    
    return pd.DataFrame(profiles)



def profile_row(row, abnormal_rate, length):
    warnings.filterwarnings('ignore')
    normal_list, abnormal_list = eval(row.normal_data), eval(row.err_data)
    combined = generate_combined_set(normal_list, abnormal_list, abnormal_rate, length)
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
    file_path = package_path + file_path
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(file_path, index=False)


# profiles = profile_data(df, abnormal_rate=0.01, length=1000)
@click.command()
@click.option('--source_data_path', prompt='Now enter the source data relative path from package path (e.g.,/source.xlsx)',
              default='/data/external/source.xlsx',
              help='source data relative path',
              show_default=True)
@click.option('--output_data_path', prompt='Now enter the output data path for training(e.g., /data/processed/profiles/1/ver_1_len_1000_rate_0.01.csv)',
              default='/data/processed/profiles/2/ver_2_len_1000_rate_0.01.csv',
              help='training data relative path from package path',
              show_default=True)
def main(source_data_path:str, output_data_path:str ):
    # load data
    df = load_excel_data(source_data_path)
    # generate and profile
    profiles_job = profile_data_joblib(df, abnormal_rate=0.01, length=1000)
    # data split into a train set and a test set
    train_df, test_df = train_test_split(profiles_job, test_size=0.1, random_state=42)
    # profiles for training
    to_csv_with_directory(train_df,  output_data_path)
    # profiles for testing
    parts = output_data_path.split('/')
    parts.insert(5, 'test')
    test_data_path = '/'.join(parts)
    to_csv_with_directory(test_df,  test_data_path)
    
if __name__ == "__main__":
    main()