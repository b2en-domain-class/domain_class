# from pydantic import BaseModel
# from typing import Dict, Any
import pandas as pd
from pandas import Series

class BuildFeatures():
    """
    ## 전처리
    - None, NULL 등의 갯수/비율 체크 및 삭제
    ##전체패턴
    - 날짜 형식 비율 
    - 숫자 비율(실수, 정수, 00012같은 것은 걸려낼 것)
    - 정수 비율
    - 번호패턴 비율
    - 문자번호패턴 비율 
    - 이메일 비율
    - URL 비율
    ##부분패턴
    - 숫자포함
    - 문자포함(한글영문)
    - discriminator포함
    - masking포함
    - 음수포함
    ## 데이터 속성
    - 데이터길이 Purity : 0~1, 1에 가까울수록 고정길이, (엔트로피를 사용해도 될 듯)
    - 데이터 Distinct 수
    - 데이터 엔트로피 : distinct 값의 분포. Noise로 인한 distinct가 크면 의미가 있는데, 계산량이 많이 들 수 있다.
    ##메타데이터
    - 컬럼도메인: 컬럼명으로부터 도메인 접미사 추출하여 해당도메인 onehot처리(도메인별 접미사 구성)
    - 데이터 타입
    """
    YYYY = r"(19|20)\d{2}"
    MM = r"(0[1-9]|1[012])"
    DD = r"(0[1-9]|[12][0-9]|3[01])"
    TM = r"\s+([01][0-9]|2[0-4]):[0-5][0-9](:[0-5][0-9])?(\s+(PM|AM))?"
    patterns = {
        "date_time": fr"^({YYYY}[-./]{MM}[-./]{DD}|{MM}[-./]{DD}[-./]{YYYY}|{DD}[-./]{YY}[-./]{YYYY})({TM})?$"
    }
    
    def __init__(self, series:Series, patterns:dict):
        na_rate = series.isna().mean()
        null_rate = series.str.strip().isin(['NULL']).mean()
        
        self.null_rate = na_rate + null_rate
        self.series = series[~series.isin(['NULL'])].dropna()  
        self.patterns = patterns
        
    def rate_matching_pattern(self,pattern:str)-> float:
        series = self.series.dropna()
        return series.str.match(pattern).mean()
    

    def get_length_purity(self)-> float:
        series = self.series.dropna().astype(str)
        ratio = series.str.len().value_counts(normalize=True)
        length_purity = (ratio * ratio).sum()
        return length_purity
    
    def profiling_patterns(self):
        features = {}
        for key, pattern in self.patterns:
            features[key] = self.rate_matching_pattern(pattern)
        features['len_purity'] = self.get_length_purity()
        features['datatype'] = self.series.dtype
        return pd.DataFrame(features)
            