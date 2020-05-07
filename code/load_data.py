import pandas as pd


def load_citibike_data(columns):
    citibike_data = None

    for month in range(1, 13):
        if month < 10:
            month_str = f'0{month}'
        else:
            month_str = f'{month}'
        print('Loading month', month_str)
        current_month = pd.read_csv(
            f'../dataset/2018{month_str}-citibike-tripdata.csv',
            usecols=columns
        )

        if citibike_data is None:
            citibike_data = current_month
        else:
            citibike_data = citibike_data.append(current_month)
    return citibike_data
