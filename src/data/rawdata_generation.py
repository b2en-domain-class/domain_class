import sys, os
from joblib import Parallel, delayed

import re, string
import pandas as pd
import random
from dotenv import load_dotenv, find_dotenv
import time
import click


# Load the .env file
load_dotenv(find_dotenv())
package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
sys.path.append(package_path)
from src.features.build_features import BuildFeatures

def load_excel_data(file_path):
    """
    :param file_path: relative path from package path
    """
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

class TableDataGenerator:
    def __init__(self, abnormal_rate, length):
        self.abnormal_rate = abnormal_rate
        self.length = length

    @staticmethod
    def is_korean_char(ch):
        return '\uAC00' <= ch <= '\uD7A3'

    @staticmethod
    def get_random_char(char_type):
        if char_type == 'digit':
            return str(random.randint(0, 9))
        elif char_type == 'lowercase':
            return random.choice(string.ascii_lowercase)
        elif char_type == 'uppercase':
            return random.choice(string.ascii_uppercase)
        elif char_type == 'korean':
            return chr(random.randint(0xAC00, 0xD7A3))

    @staticmethod
    def find_pattern(s):
        # Assuming BuildFeatures.patterns is a predefined dictionary of regex patterns
        keys = ["yn", "date_time", "number", "integer", "bunho", "email", "url"]
        patterns = {key: BuildFeatures.patterns[key] for key in keys}
        
        for key, pattern in patterns.items():
            if re.match(pattern, s, re.IGNORECASE):
                return pattern
        else:
            return None

    @staticmethod
    def modify_string(s):
        def get_char_type(ch):
            if ch.isdigit():
                return 'digit'
            elif ch.islower():
                return 'lowercase'
            elif ch.isupper():
                return 'uppercase'
            elif TableDataGenerator.is_korean_char(ch):
                return 'korean'
            else:
                return 'other'

        length = len(s)
        pattern = TableDataGenerator.find_pattern(s)
        attempts = 0

        while attempts < 1000:
            index = random.randint(0, length - 1)
            char_type = get_char_type(s[index])
            if char_type == 'other':
                attempts += 1
                continue

            new_char = TableDataGenerator.get_random_char(char_type)
            new_string = s[:index] + new_char + s[index + 1:]
            
            if pattern is None:
                return new_string

            if re.match(pattern, new_string, re.IGNORECASE):
                return new_string

            attempts += 1

        return None

    def generate_combined_set(self, normal_list, abnormal_list):
        abnormal_count = int(self.length * self.abnormal_rate) if abnormal_list else 0
        normal_count = self.length - abnormal_count

        chosen_abnormal = random.choices(abnormal_list, k=abnormal_count) if abnormal_list else []
        chosen_normal = random.choices(normal_list, k=normal_count) if normal_list else []
        
        if chosen_normal:
            chosen_normal = [self.modify_string(s) if s != 'NULL' else s 
                                            for s in chosen_normal if s]

        final_list = chosen_normal + chosen_abnormal
        random.shuffle(final_list)

        return final_list

    def process_row(self, row):
    
        normal_list, abnormal_list = eval(row.normal_data), eval(row.err_data)
        combined = self.generate_combined_set(normal_list, abnormal_list)
        return (row.domain, row.col_nm, combined)

    # Row-level Parallelization
    def save_batch_data_csv_row_parallel(self, batch_df, batch_number, out_directory):
        batch_filename = f"{package_path}/data/processed/table/row_parallel_batch_{batch_number}.csv"
        
        dir_path = package_path + out_directory
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        # 파일명 생성 (디렉토리명과 batch_number 결합)
        batch_filename = os.path.join(dir_path, f"{batch_number}.csv")
        
        start_time = time.time()
        batch_data = Parallel(n_jobs=-1)(delayed(self.process_row)(row) for row in batch_df.itertuples(index=False))
        
        
        domains, col_nms, data = zip(*batch_data)
        batch_df = pd.DataFrame({'domain': domains, 
                                 'col_nm': col_nms,
                                 'data': data})
        batch_df.to_csv(batch_filename, index=False)
        end_time = time.time()

        print("batch processing time: ", end_time - start_time)
        return None


    def save_batch_data_parquet_row_parallel(self, batch_df, batch_number, out_directory):
        # 디렉토리가 없으면 생성
        dir_path = package_path + out_directory
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        # 파일명 생성 (디렉토리명과 batch_number 결합)
        batch_filename = os.path.join(dir_path, f"{batch_number}.parquet")
    
        start_time = time.time()
        batch_data = Parallel(n_jobs=-1)(delayed(self.process_row)(row) for row in batch_df.itertuples(index=False))
            
        domains, col_nms, data = zip(*batch_data)
        batch_df = pd.DataFrame({'domain': domains, 
                                 'col_nm': col_nms,
                                 'col_values': data})
        batch_df.to_parquet(batch_filename, index=False)
        
        end_time = time.time()
        print("batch processing time: ", end_time - start_time)
        return None

    # Batch-level Parallelization for full data
    def save_full_data_parquet_row_parallel(self, df, batch_size=1000, out_directory=None):
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
      
        start_time = time.time()
        for i in range(total_batches):
            begin, end = i * batch_size, min((i + 1) * batch_size, len(df))
            self.save_batch_data_parquet_row_parallel(df.iloc[begin:end], i, out_directory)
        end_time = time.time()
        print("total processing time: ", end_time - start_time)

        return None

    # Batch-level Parallelization for full data
    def save_full_data_csv_row_parallel(self, df, batch_size=1000, out_directory=None):
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
      
        start_time = time.time()
        for i in range(total_batches):
            begin, end = i * batch_size, min((i + 1) * batch_size, len(df))
            self.save_batch_data_csv_row_parallel(df.iloc[begin:end], i, out_directory)
        end_time = time.time()
        print("total processing time: ", end_time - start_time)

        return None


@click.command()
@click.option('--source_data_path', prompt='Now enter the relative source data path from package path (e.g.,/data/source.xlsx)',
              default='/data/external/source.xlsx',
              help='source data relative path',
              show_default=True)
@click.option('--output_data_path', prompt='Now enter the output data path from package path(e.g., /data/processed/table/pqt)',
              default='/data/processed/table/pqt',
              help='output data relative directory',
              show_default=True)
@click.option('--abnormal_rate', prompt='Now enter the abnormal rate',
              default= 0.01,
              show_default=True)
@click.option('--length', prompt='Now enter the length',
              default= 1000,
              show_default=True)
def main(source_data_path, output_data_path, abnormal_rate, length):
    df = load_excel_data(source_data_path)
    processor = TableDataGenerator(abnormal_rate, length)
    processor.save_full_data_parquet_row_parallel(df, batch_size =50, out_directory=output_data_path)

if __name__ == '__main__':
    main()