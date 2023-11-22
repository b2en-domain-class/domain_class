# data generation module
import pandas as pd
import random
import string
from functools import wraps

def disturb(func):
    """
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        length, diff_length_ratio = kwargs.get('str_length', None), kwargs.get('diff_length_ratio', 0.01)
        len = length if random.random() >= diff_length_ratio else random.choice([length+1, length-1])
        kwargs['str_length'] = len
        return func(*args, **kwargs)
    return wrapper

def get_series(func):
    """
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        series_length = kwargs.get('series_length', None) or 10
        return pd.Series(func(*args, **kwargs) for _ in range(series_length))
    return wrapper

@get_series
@disturb
def generate_random_string(str_length=7, diff_length_ratio = 0, series_length=1):
    """
    """
    random_string = ''.join(random.choices(string.ascii_letters, k=str_length))
    return random_string