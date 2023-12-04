import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())
package_path = os.getenv('PACKAGE_PATH')
# package_path = '/home/dwna/projects/domain_class'
# sys.path.append(package_path)


def load_excel(file_path):
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
    
    
def load_csv(file_path):
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
    
def write_csv(df, file_path):
    file_path = package_path + file_path
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(file_path, index=False)