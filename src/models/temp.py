import click
import json
from hyperopt import hp



def show_params(ctx, param, value):
    # model_spaces 딕셔너리는 이전에 제공한 것과 동일하게 유지
    model_spaces = {
        "logistic_regression": {
            "preprocessing": ["standard", "minmax"],
            "C": (-4, 4)
        },
        "random_forest": {
            "preprocessing": ["standard", "minmax"],
            "n_estimators": [10, 50, 100, 200],
            "max_depth":[5, 10, 20, None]
        },
        "svm": {
            "preprocessing": ["standard", "minmax"],
            "C": (-4, 4),
            "gamma": (-4, 4)
        },
        "lgbm": {
            "preprocessing": ["standard", "minmax"],
            "learning_rate":  "10^(-4, 0)",
            "n_estimators":  [10, 50, 100, 200],
            "num_leaves": [15, 31, 63, 127], 
            "max_depth":  [5, 10, 20, -1]
        },
        "catboost": {
            "preprocessing": ["standard", "minmax"],
            "learning_rate": "10^(-4, 0)",
            "iterations": [10, 50, 100, 200],
            "depth": [4, 6, 8, 10]
        }
    }
    if ctx.obj is None:
        ctx.obj = {}

    if ctx.obj.get('model_name_shown'):
        return value

    if value in model_spaces:
        print(f"Parameter space for {value}: {json.dumps(model_spaces[value])}")
        ctx.obj['model_name_shown'] = True
    else:
        raise click.BadParameter(f"Invalid model name: {value}. Available models: {list(model_spaces.keys())}")
    return value

@click.command()
@click.option('--model_name', prompt='Enter the model name (e.g., logistic_regression, random_forest, svm, lgbm, catboost)',
              help='Name of the model.', callback=show_params)
@click.option('--params', prompt='Now enter the model parameters in JSON format (e.g., {"C": 1.0})',
              help='JSON string of model parameters.')
def main(model_name, params):
    # model_params를 JSON에서 Python 딕셔너리로 변환

    try:
        params = json.loads(params)
    except json.JSONDecodeError:
        raise ValueError("params must be a valid JSON string.")

    # 나머지 main 함수 로직 (이전 코드를 그대로 사용)
    print(params)
if __name__ == "__main__":
    main()