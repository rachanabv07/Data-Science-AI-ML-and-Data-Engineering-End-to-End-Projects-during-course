
# End to End Machine Learning - California Housing Prediction
A brief description of what this project does and who it's for

## Dataset
- Dataset is based on data from the 1990 California census.
- It is located at /cxldata/datasets/bootml/Housing_California_1

Below are the details of the features list for the given California Housing data set:

    1. longitude	
    2. latitude	
    3. housing_median_age	
    4. total_rooms	
    5. total_bedrooms	
    6. population	
    7. households	
    8. median_income	
    9. median_house_value	
    10. ocean_proximity

## SKills

- Python
- Pandas
- Numpy
- Matplotlib
- scikit-learn
- Data visualization
- Data Preprocessing
- Treating missing values in dataset
- Handling categorical variables
- Creating transformation pipelines
- Training Machine Learning models


## Installation
### For Data Analysis and Visualization
```python
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
```
### For Treating missing values in dataset and Handling categorical variables
```python
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.base import BaseEstimator, TransformerMixin
```
### For Splitting Data and Feature Scale the Dataset
```python
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

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
## California Housing Prediction - Graphs

![Graph](https://github.com/rachanabv07/Data-Science-AI-ML-and-Data-Engineering-End-to-End-Projects-during-course/blob/main/End%20to%20End%20Machine%20Learning%20Projects/%20Supervised%20-%20Regression%20ML%20type/California%20Housing%20Prediction/Graphs/H.PNG)

![Graph](https://github.com/rachanabv07/Data-Science-AI-ML-and-Data-Engineering-End-to-End-Projects-during-course/blob/main/End%20to%20End%20Machine%20Learning%20Projects/%20Supervised%20-%20Regression%20ML%20type/California%20Housing%20Prediction/Graphs/Housing.PNG)

![Graph](https://github.com/rachanabv07/Data-Science-AI-ML-and-Data-Engineering-End-to-End-Projects-during-course/blob/main/End%20to%20End%20Machine%20Learning%20Projects/%20Supervised%20-%20Regression%20ML%20type/California%20Housing%20Prediction/Graphs/Hu.PNG)

## Acknowledgements

 - End to End ML Guided Project - California Housing Prediction from CloudxLab. [https://cloudxlab.com/assessment/playlist-intro/1275/end-to-end-ml-project-beginner-friendly?course_id=165&playlist_id=1275]
