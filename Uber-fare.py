"""
The objective of this analysis is to:
1. Build Regression models to predict the fare price of uber ride in New York using one feature (independent variable), Distance travel
2. Evaluate the models and compare their respective scores like Coefficient of Determination (R2), Root Mean Square Error, etc
"""
#Data on Kaggle: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset?select=uber.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from My_functions import haversine_py #importing haversine_py function from my_functions module to convert coordinates to distance
#Importing Scikit-learn libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



raw_uber_data = pd.read_csv("uber.csv") #Import raw, uncleaned uber data
#raw_uber_data.info() #Getting infomaation about data

raw_uber_data_cd = raw_uber_data.drop(['Unnamed: 0', 'key', 'pickup_datetime'], axis=1) #dropped labels not useful for the analysis

#--EXPLORATORY ANALYSIS--
raw_uber_data.describe() #statistical description of the raw data
raw_uber_data_cd[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].min() #checked for the lowest coordinate
raw_uber_data_cd[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].max() #checked for the maximum coordinate

#--Vizualization of coordinates
raw_uber_data_cd[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].plot.box() #To determine its distribution and check for outliers

#--DATA WRANGLING--
#--Cleaning by coordinates
raw_uber_data_cd[((raw_uber_data_cd[['pickup_longitude', 'dropoff_longitude']]<-75.5) |
                  (raw_uber_data_cd[['pickup_longitude','dropoff_longitude']]>-72.5))] = np.nan #assigning null value to extreme longitudes
raw_uber_data_cd[((raw_uber_data_cd[['pickup_latitude', 'dropoff_latitude']]<39.5) |
                  (raw_uber_data_cd[['pickup_latitude','dropoff_latitude']]>41.5))] = np.nan #assigning null value to extreme latitudes

#--Conversion of coordinates to distance--
raw_uber_data_cd['distance_in_miles'] = haversine_py(raw_uber_data_cd['pickup_latitude'],
                                                 raw_uber_data_cd['pickup_longitude'], raw_uber_data_cd['dropoff_latitude'],
                                                 raw_uber_data_cd['dropoff_longitude'], unit="mile")

raw_uber_data_cd[((raw_uber_data_cd['distance_in_miles']<0.3) |
                  (raw_uber_data_cd['distance_in_miles']>30))] = np.nan #nulifying all distances of 0.3 miles and below

#-- cleaning fare_amount
raw_uber_data_cd['fare_amount'].describe()
raw_uber_data_cd[((raw_uber_data_cd['fare_amount']<=2) |
                  (raw_uber_data_cd['fare_amount'] > 300))] = np.nan #assigning null value to extreme fare amount

raw_uber_data_drop_dup = raw_uber_data_cd.drop_duplicates() #drop duplicates

cleaned_uber_data = raw_uber_data_drop_dup.dropna().copy() #a cleaned copy of the original raw data

#--Exploring relationships between labels
cleaned_uber_data[['distance_in_miles', 'fare_amount']].plot.scatter(x='distance_in_miles', y='fare_amount') #visualizing the relationship between distance and fare amount
cleaned_uber_data['distance_in_miles'].corr(cleaned_uber_data['fare_amount']) #0.90 - relationship between fare amount and the distance in miles
cleaned_uber_data['passenger_count'].corr(cleaned_uber_data['fare_amount']) #0.02 - relationship between fare amount and passanger count


#--PREDICTIVE ANALYSIS--
#--Data preprocessing
X = cleaned_uber_data['distance_in_miles'].values[:, np.newaxis] #reshaping X [[1.047], [1.528], [3.131]...[]]
y = cleaned_uber_data['fare_amount'].values

#--Linear regression model and prediction
model = LinearRegression()

grid_estimator = GridSearchCV(model, param_grid={'n_jobs': [-1]}, cv=4)
grid_estimator.fit(X,y)

predicted_y =  grid_estimator.predict(X) #predicting y using X

#--Linear regression metrics
R2 = r2_score(y, predicted_y) #0.81
mean_ae = mean_absolute_error(y, predicted_y) #2.25
mean_se = mean_squared_error(y, predicted_y) #16.91
rms_error = np.sqrt(mean_se) #4.11

print(f"""
Coefficient of determination = {R2:.2f}
Mean absolute error = {mean_ae:.2f}
Root mean squared error = {rms_error:.2f}""")

#--KNeighbor regressor model and prediction
k_neighbor = KNeighborsRegressor()
neighbor_grids = GridSearchCV(estimator=k_neighbor, param_grid={'n_neighbors': [10], 'n_jobs': [-1]}, cv=3)
"""
n_neighbors: [10] seems to give the best mean test score, having tried [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] nearest neighbors
"""
neighbor_grids.fit(X,y) #fitting the model
df = pd.DataFrame(neighbor_grids.cv_results_) #DataFrame to view test scores
pred_y = neighbor_grids.predict(X)

#--KNeighbor metrics
R_2 = r2_score(y, pred_y) #0.84
mae = mean_absolute_error(y, pred_y) #2.15
mse = mean_squared_error(y, pred_y)
rmse = np.sqrt(mean_se) #4.11

print(f"""
Coefficient of determination = {R_2:.2f}
Mean absolute error = {mae:.2f}
Root mean squared error = {rmse:.2f}""")


#---COMPARISON---
#Both LinearRegression() and KNeighborsRegressor() give similar predictions
#Coefficient of Determination: LinearRegression() = 0.81 | KNeighborsRegressor() = 0.84
#Mean Absolute Error: LinearRegression() = 2.25 | KNeighborsRegressor() = 2.15
#Root Mean Squared Error: LinearRegression() = 4.11 | KNeighborsRegressor() = 4.11
