import logging

import pandas as pd

logger = logging.getLogger(__name__)


def citibike(columns: list) -> pd.DataFrame:
    """
    Loads the entire year 2018 from the citibike dataset.

    :param columns: preselects some columns during loading
    :return: the citibike dataset as a dataframe
    """
    citibike_data = None

    for month in range(1, 13):
        if month < 10:
            month_str = f'0{month}'
        else:
            month_str = f'{month}'
        logger.info(f'Loading month {month_str}...',)
        current_month = pd.read_csv(
            f'../dataset/2018{month_str}-citibike-tripdata.csv',
            usecols=columns
        )

        if citibike_data is None:
            citibike_data = current_month
        else:
            citibike_data = citibike_data.append(current_month)
    return citibike_data
