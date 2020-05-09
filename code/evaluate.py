from code import store_model
from code.config import config


def run():
    """
    Prints the metrics for the citibike model
    """
    print(config)

    model_data = store_model.load_model_data()

    print('F1 Score:', model_data['metadata']['f1_score'])
    print(model_data['metadata']['classification_report'])


if __name__ == '__main__':
    run()
