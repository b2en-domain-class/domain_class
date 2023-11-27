# build_feature.py
import pandas as pd
import numpy as np
import re
from pandas import Series
import warnings
warnings.filterwarnings('ignore')

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
    
    # usage
        data = {
            'sample_column': [
                '2023-01-01', '100', '300.5', 'test@example.com', 'http://example.com', 
                '2023-02-01', '200', '400.5', 'hello@world.com', 'https://world.com',
                '2023-03-01', 'NULL', None, 'NaN', '2023-04-01'
            ]
        }
        # 판다스 데이터프레임 생성
        df = pd.DataFrame(data)
        # BuildFeatures 클래스 인스턴스화
        bf = BuildFeatures(df['sample_column'], 'SAMPLE_sz')
        # 프로파일링 패턴 호출
        profile = bf.profiling_patterns()
        print(profile)    
            
    """
    YYYY = r"(19|20)\d{2}"
    MM = r"(0[1-9]|1[012])"
    DD = r"(0[1-9]|[12][0-9]|3[01])"
    TM = r"\s+([01][0-9]|2[0-4]):[0-5][0-9](:[0-5][0-9])?(\s+(PM|AM))?"
    patterns = {
        "date_time": fr"^({YYYY}[-./]{MM}[-./]{DD}|{MM}[-./]{DD}[-./]{YYYY}|{DD}[-./]{MM}[-./]{YYYY})({TM})?$",
        "number": r"^(?!0\d)\d+([.]\d*)?$",
        "integer": r"^(?!0\d)\d+$",
        "bunho": r"^[A-Za-z0-9\uAC00-\uD7A3]+([-./:][A-Za-z0-9\uAC00-\uD7A3]+)*$",
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$",
        "part_num": r"\d+",
        "part_text": r"[A-Za-z\uAC00-\uD7A3]+",
        "part_discriminator": r"[./-:]",
        "part_mask": r"[#*]{3,}",
        "part_minus": r"^-\d",
    }

    #  도메인 접미사 패턴 그룹 정의 및 결합
    suffix_patterns = {
        "BUNHO": "(?P<BUNHO>(NO|SN|ZIP|TKN|VIN|ENTN)$)",
        "NALJJA": "(?P<NALJJA>(DT|YMD|YM|YR|SYR|MM|DAY)$)",
        "MYEONG": "(?P<MYEONG>(NM|TTL)$)",
        "JUSO": "(?P<JUSO>(ADDR)$)",
        "YEOBU": "(?P<YEOBU>YN$)",
        "CODE": "(?P<CODE>CD$)",
        "ID": "(?P<ID>ID$)",
        "SURYANG": "(?P<SURYANG>(LOT|DONT|GRD|LVL|GFA|AREA|PRG|SCR|CNT|LEN|SZ)$)",
        "GEUMAEK": "(?P<GEUMAEK>(AMT|FEE|UNTPRC)$)",
        "NAEYOUNG": "(?P<NAEYOUNG>CN$)",
        "YUL": "(?P<YUL>RT$)",
    }
    combined_pattern = re.compile("|".join(suffix_patterns.values()),  re.IGNORECASE)
    
    def __init__(self, series:Series, col_name:str=None, domain:str=None):
        if not isinstance(series, pd.Series):
            raise ValueError("series must be a pandas Series")        
        
        na_rate = series.isna().mean()
        null_rate = series.str.strip().isin(['NULL', 'NaN']).mean()
        
        self.null_rate = na_rate + null_rate
        self.series = series[~series.isin(['NULL', 'NaN'])].dropna()
        self.datatype = self.series.dtype.name  
        self.col_name = col_name if col_name else self.series.name
        self.domain = domain

        # self.patterns = patterns
        
    def rate_matching_pattern(self, pattern:str)-> float:
        # series = self.series.dropna()
        return self.series.str.contains(pattern, regex=True).mean()
    

    def get_length_purity(self)-> float:
        # series = self.series.dropna().astype(str)
        ratio = self.series.str.len().value_counts(normalize=True)
        length_purity = (ratio * ratio).sum()
        return length_purity
    
    def get_value_nunique(self):
        return self.series.nunique()
        # self.series.value_counts()
        
    def get_value_distr(self):
        ratio = self.series.value_counts(normalize=True)
        entropy = (-ratio * np.log2(ratio)).sum()
        return entropy
    
    def find_suffix_domain(self):
        """
        컬럼명에서 도메인분류어 패턴을 추출하여 어떤 도메인그룹의 suffix에 해당하는지 검출
        :return: 매칭된 그룹 이름 또는 '해당 사항 없음'
        """
        word = self.col_name
        if not isinstance(word, str) or not word:
            return "유효하지 않은 입력"
        
        match = self.combined_pattern.search(word)
        if match:
            matched_groups = [name for name, value in match.groupdict().items() if value]
            return matched_groups if matched_groups else "ETC"
        else:
            return "ETC"
           
    def one_hot_encode(self, domain:str, categories: list) -> dict:
       """
       주어진 pandas Series를 one-hot 인코딩으로 변환합니다.
    
       :param series: 인코딩할 pandas Series 객체
       :param categories: one-hot 인코딩을 위한 카테고리 리스트
       :return: 카테고리 값을 키로 하고 one-hot 인코딩된 pd.Series를 값으로 하는 딕셔너리
       """
       if not isinstance(categories, list):
           categories = list(categories)
       categories = categories +['ETC'] # categories에 들어가지 않는 것은 ETC로 추가로 분류한다.
       one_hot_encoded = pd.get_dummies(domain, prefix='', prefix_sep='').reindex(columns=categories, fill_value=0)
       return one_hot_encoded.iloc[0].to_dict() 

    def profiling_patterns(self)-> Series:
        features = {'col_name':self.col_name}
        for key, pattern in self.patterns.items():
            features[key] = self.rate_matching_pattern(pattern)
        features['len_purity'] = self.get_length_purity()
        features['value_nunique'] = self.get_value_nunique()
        features['value_distr'] = self.get_value_distr()
        features['datatype'] = self.datatype
        suffix_domain = self.find_suffix_domain()
        one_hot_encoded = self.one_hot_encode(suffix_domain, self.suffix_patterns.keys()) 
        features = {**features, **one_hot_encoded}
        if self.domain:
            features['domain'] = self.domain
        return pd.Series(features)       # return pd.Series(features,name = self.col_name )

    
### use_case
# data = {
#     'sample_column': [
#         '2023-01-01', '100', '300.5', 'test@example.com', 'http://example.com', 
#         '2023-02-01', '200', '400.5', 'hello@world.com', 'https://world.com',
#         '2023-03-01', 'NULL', None, 'NaN', '2023-04-01'
#     ]
# }


# data = {

#     'ENTN': [
#         '200905830','201785062','080146287','200100957','201810936'
#     ]
# }

# # 판다스 데이터프레임 생성
# df = pd.DataFrame(data)
# # BuildFeatures 클래스 인스턴스화
# bf = BuildFeatures(df['ENTN'], 'ENTN')
# # 프로파일링 패턴 호출
# profile = bf.profiling_patterns()
# print(profile)    
