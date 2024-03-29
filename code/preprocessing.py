import numpy as np
import pandas as pd
from sklearn import preprocessing

from code.config import config


def _select_balanced_sample(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.groupby(config['dataset']['label_column']).apply(lambda x: x.sample(500000))
    dataset = dataset.reset_index(level=[0,1], drop=True)
    return dataset


def _fix_tripduration(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes tripdurations that do not match the start & stop times.

    :param dataset: the citibike dataset containing start & stoptime
    :return: the citibike set with fixed tripduration
    """
    dataset['starttime'] = pd.to_datetime(dataset['starttime'])
    dataset['stoptime'] = pd.to_datetime(dataset['stoptime'])

    dataset['tripduration'] = (dataset['stoptime'] - dataset['starttime'])
    dataset['tripduration'] = dataset['tripduration'].dt.total_seconds().apply(np.floor).astype(int)
    return dataset


def _filter_columns(dataset: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Filters out certain columns which are not required for the prediction.

    :param dataset: the citibike dataset
    :return: the reduced citibike dataset
    """
    filter_columns = config['dataset']['filter_columns']
    if not training and config['dataset']['label_column'] in filter_columns:
        filter_columns.remove(config['dataset']['label_column'])

    return dataset[filter_columns]


def _remove_duration_outlier(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Removes some outliers with very large tripdurations (> 20 days).

    :param dataset: the citibike dataset with the tripduration column
    :return: the citibike dataset without trips longer than 20 days
    """
    return dataset[dataset['tripduration'] <= 20 * 24 * 60 * 60]


def _remove_geo_outlier(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers in the geo location using a bounding box around the NYC area.

    :param dataset: the citibike dataset containing geo locations for start & end stations
    :return: the citibike dataset containing only geo locations in the NYC area
    """
    dataset = dataset[dataset['end station latitude'] <= 41.13]
    dataset = dataset[dataset['start station latitude'] <= 41.13]
    dataset = dataset[dataset['end station latitude'] >= 40.12]
    dataset = dataset[dataset['start station latitude'] >= 40.12]
    dataset = dataset[dataset['end station longitude'] >= -74.40]
    dataset = dataset[dataset['start station longitude'] >= -74.40]
    dataset = dataset[dataset['end station longitude'] <= -73.19]
    dataset = dataset[dataset['start station longitude'] <= -73.19]
    return dataset


def _create_weekday_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a weekday from the trip starttime.

    :param dataset: the citibike dataset containing the starttime column
    :return: the citibike dataset containing a new start_weekday column
    """
    dataset['start_weekday'] = dataset['starttime'].dt.day_name()
    return dataset


def _transform_to_datetime(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the startime to a datetime

    :param dataset: the citibike dataset containing the starttime column
    :return: the citibike dataset with transformed starttime
    """
    dataset['starttime'] = pd.to_datetime(dataset['starttime'])
    return dataset


def _encode_weekday(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Label encodes the weekdays.
    :param dataset: the citibike dataset containing the start_weekday column
    :return: the citibike dataset with the encoded start_weekday column
    """
    unique_weekdays = dataset['start_weekday'].unique()
    weekday_encoder = preprocessing.LabelEncoder()
    weekday_encoder.fit(unique_weekdays)
    dataset['start_weekday'] = weekday_encoder.transform(dataset['start_weekday'])
    return dataset


def encode_usertype(dataset: pd.DataFrame):
    """
    Encodes the usertype.
    :param dataset: the citibike dataset containing the usertype column
    :return: the citibike dataset with the encoded usertype column and the encoder
    """
    label_column = config['dataset']['label_column']

    unique_labels = dataset[label_column].unique()
    usertype_encoder = preprocessing.LabelEncoder()
    usertype_encoder.fit(unique_labels)
    dataset[label_column] = usertype_encoder.transform(dataset[label_column])
    return dataset, usertype_encoder


def run(dataset: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """
    performs all the preprocessing steps for the citibike data.

    :param dataset: the citibike dataset
    :param training: is the preprocessing performed for training or prediction
    :return: the preprocessed dataset
    """
    if training:
        dataset = dataset.dropna()
        dataset = _fix_tripduration(dataset)
        dataset = _remove_duration_outlier(dataset)
        dataset = _remove_geo_outlier(dataset)
        dataset = _select_balanced_sample(dataset)
    else:
        dataset = _transform_to_datetime(dataset)

    dataset = _create_weekday_feature(dataset)
    dataset = _encode_weekday(dataset)
    dataset = _filter_columns(dataset, training)
    return dataset
