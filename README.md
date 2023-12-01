# 데이터베이스 컬럼 데이터 도메인 추정을 위한 머신러닝 모델 개발 및 성능 평가

## 소개
이 연구는 데이터베이스에서 추출한 열 데이터의 도메인을 자동으로 추정하기 위한 머신러닝 모델을 개발하고 성능을 평가하는 것에 초점을 맞추고 있습니다. 데이터베이스 관리 및 데이터 분석에서 열의 도메인을 식별하는 것은 데이터의 유효성, 일관성 및 적절한 처리 방법을 결정하는데 중요합니다. 이 프로젝트는 이러한 도전을 해결하기 위한 자동화된 접근 방식을 제시하며, 다양한 머신러닝 알고리즘과 심층적인 데이터 분석을 사용합니다.

## 모델 설명
이 모델은 열 데이터 패턴, 메타데이터 및 데이터 속성에서 파생된 종합적인 기능 세트를 사용합니다. 기능 추출에는 패턴 매칭(전체 및 부분), 데이터 속성(예: 엔트로피 및 구별성) 및 메타데이터 분석(특히 도메인 접미사 추출 및 원-핫 인코딩에 중점을 둠)이 포함됩니다. 사용된 머신러닝 알고리즘은 로지스틱 회귀, 랜덤 포레스트, SVM, LightGBM 및 CATBOOST이며, 각각 분류 작업에서의 고유한 강점을 위해 선택되었습니다. `sklearn.metrics`의 성능 지표는 다양한 도메인 카테고리에서 모델의 효과를 보여줍니다.

## 데이터셋 설명
열 데이터를 기반으로 데이터셋이 생성되었으며, 관련 기능을 추출하기 위해 사용자 정의 모듈이 개발되었습니다. 모델 평가를 위해 별도의 테스트 데이터셋이 준비되었습니다. 하이퍼파라미터는 훈련 데이터셋에서만 교차 검증 기술을 사용하여 최적화되었으며, 정확도를 향상시키기 위해 모델을 정제했습니다.

## 설치 및 의존성
- **파이썬 버전**: 3.8.10
- **의존성**: 필요한 패키지는 `requirements.txt` 파일에서 참조하세요.
- **MLflow**: 모델 관리 및 추적을 위해 사용됩니다. 사용 섹션에서 제공된 설정 지침을 따르세요.

## 사용 방법
1. **MLflow 추적 서버 설정**: 
   ```bash
   mlflow server --backend-store-uri sqlite:////var/mlflow/mlflow.db --default-artifact-root /path/to/artifacts --host 0.0.0.0 --port 5000
   ## Generate Training Data
2. **훈련데이터 생성**:
   ```bash
   python -m make_trainingdataset --source_data_path '/source.xlsx' --ouptput_data_path '/1/train.csv'
3. **하이퍼파라미터 최적화**:
   ```bash
   python -m hype_opt model_name train_data_path
4. **모델 훈련**:
   ```bash
   python -m train_model model_name params train_data_path
5. **모델 적용**:
   ```bash
   python -m domain_estimate model file.csv


## 결과 및 시각화
모델은 전반적으로 88.24%의 정확도를 달성하며, 날짜, 번호 및 여부와 같은 카테고리에서 강력한 성능을 보여줍니다. 각 카테고리의 세부 성능 지표는 모델의 정밀도, 재현율 및 F1 점수를 강조하여 제공됩니다.

## 기여
기여는 환영합니다! 코드 개선, 버그 수정 또는 문서 개선을 통해 기여할 수 있습니다. 이 섹션에 설명된 대로 풀 리퀘스트 및 문제 보고에 대한 지침을 따라주세요.

## 라이센스
이 프로젝트는 MIT 라이센스에 따라 라이센스가 부여됩니다. 저작권 고지 및 면책 조항이 포함되어 있으면 프로젝트를 사용, 수정 및 배포하는 데 넓은 자유가 허용됩니다.

## 연락처 정보
질문, 제안 또는 잠재적 협업 기회에 대해 [dwna@b2en.com](mailto:dwna@b2en.com)으로 이메일을 보내주세요.


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
            │                 models to make  predictions
            ├── domain_estimate.py
            ├── train_model.py
            └── hype_opt.py

     
    


--------
