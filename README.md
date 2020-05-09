# Citibike Data Analysis

Analysis of the citibike (https://www.citibikenyc.com/)

* Explore the citibike data (for the year 2018)
* Classify customers as (`Customer` or `Subscriber`)


Download data at: https://s3.amazonaws.com/tripdata/index.html
More information about the dataset: https://www.citibikenyc.com/system-data

Download the files into a folder called `dataset` with the following naming scheme: `2018{month}-citibike-tripdata.csv`

Extract the model in the `model` directory with path `model/citibike_model.pkl`

To build the project execute:
```console
make base
```

To run the small API:
```console
make api
```

Some sample requests:
```console
curl 127.0.0.1:8080/status

curl 127.0.0.1:8080/predict -H "Content-Type: application/json" -X POST -d '{"tripduration": 1200, "starttime": "2020-05-09 13:30:00.000", "start_station_latitude": 40.794067, "start_station_longitude": -73.962868, "end_station_latitude": 40.764397, "end_station_longitude": -73.973715, "birth_year": 1970, "gender": 0}'

curl 127.0.0.1:8080/predict -H "Content-Type: application/json" -X POST -d '{"tripduration": 1200, "starttime": "2020-05-09 13:30:00.000", "start_station_latitude": 40.794067, "start_station_longitude": -73.962868, "end_station_latitude": 40.764397, "end_station_longitude": -73.973715, "birth_year": 1969, "gender": 0}'
```