import pickle as pkl
from pathlib import Path

model_filepath = Path(__file__).parents[0].resolve() / Path('../model/citibike_model.pkl')


def save_model_data(model_data):
    print('Saving model data...')
    with open(model_filepath, 'wb+') as file_handle:
        pkl.dump(model_data, file_handle)


def load_model_data():
    print('Saving model data...')
    with open(model_filepath, 'rb') as file_handle:
        return pkl.load(file_handle)
