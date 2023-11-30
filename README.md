domain_class
==============================


이 프로그램은 데이터베이스의 품질을 관리하고 개선함에 있어서,
데이터의 정확성, 일관성 및 신뢰성을 보장함으로써, 데이터 중심의 비즈니스 의사결정에 도움을 줍니다

도메인 추정 기능을 통해, 데이터베이스 내 각 필드의 데이터 유형과 패턴을 자동으로 식별하고 분석합니다. 이를 통해 데이터베이스의 구조와 내용에 대한 깊은 이해를 얻을 수 있습니다.

머신러닝 알고리즘을 이용하여 도메인 추정하기 때문에, 학습데이터 축적에 따른 지속적인 개선이 가능합니다.

데이터프레임 형식의 컬럼에 해당하는 데이터셋의 도메인을 추정하는 알고리즘입니다.

### use_case
```python
data = {
    'sample_column': [
        '2023-01-01', '100', '300.5', 'test@example.com', 'http://example.com', 
        '2023-02-01', '200', '400.5', 'hello@world.com', 'https://world.com',
        '2023-03-01', 'NULL', None, 'NaN', '2023-04-01'
    ]
}
#판다스 데이터프레임 생성   
df = pd.DataFrame(data)   

#BuildFeatures 클래스 인스턴스화   
bf = BuildFeatures(df['sample_column'], 'SAMPLE_sz')   
#프로파일링 패턴 호출   
profile = bf.profiling_patterns()   
print(profile)    
```

이를 위해 정규표현식을 이용한 다양한 패턴들, 데이터의 값 및 특성의 분포을 이용해 Feature를 구성하였습니다.

Feature 구성을 위해 사용된 패턴 및 분포   

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
  
    YYYY = r"(19|20)\d{2}"
    MM = r"(0[1-9]|1[012])"
    DD = r"(0[1-9]|[12][0-9]|3[01])"
    TM = r"\s+([01][0-9]|2[0-4]):[0-5][0-9](:[0-5][0-9])?(\s+(PM|AM))?"
    patterns = {
        "date_time": fr"^({YYYY}[-./]{MM}[-./]{DD}|{MM}[-./]{DD}[-./]{YYYY}|{DD}[-./]{MM}[-./]{YYYY})({TM})?$",
        "number": r"^(?!0\d)\d+([.]\d*)?$",
        "integer": r"^(?!0\d)\d+$",
        # "bunho": r"^\d+([-./]\d+)*$",
        "bunho": r"^[A-Za-z0-9\uAC00-\uD7A3]+([-./:][A-Za-z0-9\uAC00-\uD7A3]+)*$",
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$",
        "part_num": r"\d+",
        "part_text": r"[A-Za-z\uAC00-\uD7A3]+",
        "part_discriminator": r"[./-:]",
        "part_mask": r"[#*]{3,}",
        "part_minus": r"^-\d",
    }
    #  도메인 접미사 패턴 
    suffix_patterns = {
        "BUNHO": "(?P<BUNHO>(NO|SN|ZIP|TKN|VIN)$)",
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


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to generate data
        │   └── make_train_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to hyperoptimize the parameters of models, train models and then use trained  
        │   │                 models to make  predictions
        │   ├── predict_model.py
        │   ├── train_model.py
        │   └── hype_opt.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
     
    


--------



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
