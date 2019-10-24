# -*- coding: utf-8 -*-
"""
@author: Jessica Cabral
"""
############################################################################
#       Data Preparation Script
############################################################################

import sys
import os
import pandas as pd
import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.decomposition import PCA

##############################
#    Default paths
##############################
PATH_RAW_DATA = '../data/raw/'
#PATH_RAW_DATA = 'data/raw/'
PATH_PROCESSED_DATA = '../data/processed/'

print("Path Raw Data: {}".format(PATH_RAW_DATA))
print("Path Processed Data: {}".format(PATH_PROCESSED_DATA))

##############################
#     Data read
##############################
print('Reading Data...')
bikes = pd.read_csv(PATH_RAW_DATA+'bikes.csv')
print('Data Shape: {}\n'.format(bikes.shape))

##############################
#     PCA
##############################
#pca=PCA(n_components=1)
#pca.fit(bikes[['temp', 'atemp']])
#bikes['temp_PCA']=pca.fit_transform(bikes[['temp','atemp']])

##############################
#     transform the "dteday" feature to date type
##############################
#bikes["dteday"] = pd.to_datetime(bikes["dteday"])

##############################
#     Feature Engineering
##############################
date = pd.DatetimeIndex(bikes['dteday'])

bikes['year'] = date.year
#bikes['month'] = date.month
#bikes['hour'] = date.hour
bikes['dayofweek'] = date.dayofweek

bikes['year_season'] = bikes['year'] + bikes['season'] / 10

bikes['hour_workingday_casual'] = bikes[['hr', 'workingday']].apply(
        lambda x: int(10 <= x['hr'] <= 19), axis=1)

bikes['hour_workingday_registered'] = bikes[['hr', 'workingday']].apply(
        lambda x: int((x['workingday'] == 1 and (x['hr'] == 8 or 17 <= x['hr'] <= 18))
        or (x['workingday'] == 0 and 10 <= x['hr'] <= 19)), axis=1)
    
by_season = bikes.groupby('year_season')[['cnt']].median()
by_season.columns = ['count_season']

bikes = bikes.join(by_season, on='year_season')
##############################
#     Replace windwindspeed
##############################
#bikes.loc[bikes['windspeed']==0, 'windspeed'] = bikes['windspeed'].mean()


##############################
#     One-Hot-Encoding
##############################
print("One-Hot-Encoding....")
def dummify_dataset(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df = df.drop([column], axis=1)
    return df


columns_to_dummify = ['season', 'weathersit', 'mnth'] 
for column in columns_to_dummify:
    bikes = dummify_dataset(bikes, column)
    
print(bikes.head(1))

##############################
#     Normalize features - scale
##############################
print('Feature Normalizarion...')
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr'] # weekday, daysofweek, count_season

print('Features before the normalization')
print(bikes.loc[:, numerical_features][:5])

# Normalizing...
bikes.loc[:, numerical_features] = preprocessing.scale(bikes.loc[:, numerical_features])

print('Features after the normalization')
print(bikes.loc[:, numerical_features][:5])

##############################
#     Log Target
##############################
#bikes['cnt'] = np.log(bikes['cnt']+1)

##############################
#     Selecting features that we are going to use
##############################
print('Selection Features...')
features_to_be_removed = ['dteday', 'casual', 'registered', 'instant']
bikes = bikes[bikes.columns.difference(features_to_be_removed)]
print('Data Shape after feature selection: {}\n'.format(bikes.shape))

##############################
#     Save processed Dataset
##############################
print('\nSaving...')
if not os.path.exists(PATH_PROCESSED_DATA):
    os.makedirs(PATH_PROCESSED_DATA)

bikes.to_csv(PATH_PROCESSED_DATA+'bikes_processed.csv',index=False, encoding='utf-8-sig')

print('\nBike Processed saved with sucess!')
