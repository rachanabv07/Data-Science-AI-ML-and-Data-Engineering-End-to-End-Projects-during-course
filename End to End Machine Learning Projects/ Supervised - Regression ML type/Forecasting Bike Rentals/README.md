
# End to End Machine Learning Projects - Forecasting Bike Rentals

## Objective

- The objective of the project is - using historical usage patterns and weather data, forecast(predict) bike rental demand (number of bike users (‘cnt’)) on hourly basis.

- Use the provided “Bikes Rental” data set to predict the bike demand (bike users count - 'cnt') using various best possible models (ML algorithms). Also, report the model that performs best, fine-tune the same model using one of the model fine-tuning techniques, and report the best possible combination of hyperparameters for the selected model. Lastly, use the selected model to make final predictions and compare the predicted values with the actual values.

Below are the details of the features list for the given Bikes data set:

    1. instant: record index

    2. dteday : date

    3. season: season (1: springer, 2: summer, 3: fall, 4: winter)

    4. yr: year (0: 2011, 1:2012)

    5. mnth: month (1 to 12)

    6. hr: hour (0 to 23)

    7. holiday: whether the day is a holiday or not

    8. weekday: day of the week

    9. workingday: if day is neither weekend nor holiday is 1, otherwise is 0.

    10. weathersit:

        - Clear, Few clouds, Partly cloudy, Partly cloudy

        - Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

        - Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

        - Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

    11. temp: Normalized temperature in Celsius. The values are derived via (tt_min)/(t_maxt_min), t_min=*8, t_max=+39 (only in hourly scale)

    12. atemp: Normalized feeling temperature in Celsius. The values are derived via (tt_min)/(t_maxt_min), t_min=*16, t_max=+50 (only in hourly scale)

    13. hum: Normalized humidity. The values are divided to 100 (max)

    14. windspeed: Normalized wind speed. The values are divided to 67 (max)

    15. casual: count of casual users

    16. registered: count of registered users

    17. cnt: count of total rental bikes including both casual and registered users

## Steps followed to solve the problem

    1. Importing the libraries

    2. Using some pre-defined utility functions

    3. Loading the data

    4. Cleaning the data

    5. Dividing the dataset into training and test dataset

        - using train_test_split in the ratio 70:30
    6. Training several models and analyzing their performance to select a model

    7. Fine-tuning the model by finding the best hyper-parameters and features

    8. Evaluating selected model using test dataset

## SKills

- Pandas
- Regression
- Machine Learning
- Python Programming
- Data Preprocessing
- scikit-learn
- Matplotlib




## Installation
### For Data Analysis and Visualization
```python
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
```
### For Splitting Data and Feature Scale the Dataset
```python
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler

```
### For Training Data and Analysis
```python
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
```
### For Fine Tune the Model
```python
    from sklearn.model_selection import GridSearchCV
```
## Bike Demand (predicted values v/s actual values)

![Graph](https://github.com/rachanabv07/Data-Science-AI-ML-and-Data-Engineering-End-to-End-Projects-during-course/blob/main/End%20to%20End%20Machine%20Learning%20Projects/%20Supervised%20-%20Regression%20ML%20type/Forecasting%20Bike%20Rentals/Images/BDA.PNG)

![Graph](https://github.com/rachanabv07/Data-Science-AI-ML-and-Data-Engineering-End-to-End-Projects-during-course/blob/main/End%20to%20End%20Machine%20Learning%20Projects/%20Supervised%20-%20Regression%20ML%20type/Forecasting%20Bike%20Rentals/Images/BDP.PNG)

## Acknowledgements

 - End to End ML Guided Project - Bikes Assessment from CloudxLab. [https://cloudxlab.com/assessment/displayslide/2327/end-to-end-project-bikes-assessment-basic-description?course_id=165&playlist_id=187]


