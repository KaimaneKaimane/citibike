from code.load_data import load_citibike_data

extract_columns = [
    'tripduration',
    'starttime',
    'stoptime',
    'start station id',
    'start station latitude',
    'start station longitude',
    'end station id',
    'end station latitude',
    'end station longitude',
    'usertype',
    'gender'
]
load_citibike_data()