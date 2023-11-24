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

file_path = "quality_evaluation_data.xlsx"
df_e = pd.read_excel(file_path, sheet_name='error_samples')
df_n = pd.read_excel(file_path, sheet_name='normal_samples')