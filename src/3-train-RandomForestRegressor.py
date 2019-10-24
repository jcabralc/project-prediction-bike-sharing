# -*- coding: utf-8 -*-
"""
@author: Jessica Cabral
"""
############################################################################
#       Train Script
############################################################################


import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

#mlflow.delete_experiment('0')
#mlflow.delete_run('094a317d4cd342539339cdae26951efd')

np.random.seed(40)

try:
    import cPickle as pickle
except ImportError:
    import pickle

##############################
#    Default paths
##############################
PATH_PROCESSED_DATA = '../data/processed/'
PATH_MODEL = '../models/'

if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)

###############################
##    Default Variables
###############################
    
# Test data set split ratio
split = 0.33
seed = 201910

algorithm_name = 'RandomForestRegressor'
###############################
##  Function Evaluation
###############################
#def RMSLE_metric(real, predicted):
#    sum=0.0
#    for x in range(len(predicted)):
#        if predicted[x]<0 or real[x]<0: #check for negative values
#            continue
#        p = np.log(predicted[x]+1)
#        r = np.log(real[x]+1)
#        sum = sum + (p - r)**2
#    return (sum/len(predicted))**0.5

def RMSLE_metric(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

def eval_metrics(actual, pred):
    MSLE = mean_squared_log_error(actual, pred) 
    MSE = mean_squared_error(actual, pred) 
    R2 = r2_score(actual, pred)  
    MAE = mean_absolute_error(actual, pred)
    RMSLE = RMSLE_metric(actual, pred)
    return R2, MAE, RMSLE , MSE,MSLE 

###############################
##    Model Parameters
###############################
#max_depth = float(sys.argv[1]) # if len(sys.argv) > 1 else 0.5
#n_estimators = int(sys.argv[2]) # if len(sys.argv) > 2 else 0.5

models_params = { 
        'max_depth': 20,
        'n_estimators': 150}
              
###############################
##    Create Folder model version
###############################
try:
    model_version = len(next(os.walk(PATH_MODEL))[1])
    if model_version is None:
        model_version = 0
except StopIteration :
    model_version = 0
    
model_version_name = 'modelv{}-{}'.format(model_version+1, datetime.now().strftime("%d%m%Y-%H%M%S"))
model_version_path = r'{}/{}'.format(PATH_MODEL,model_version_name) 

if not os.path.exists(model_version_path):
    os.makedirs(model_version_path)

print('Versao do modelo: {}'.format(model_version_name))
###############################
##    Create Folder model version
###############################
# Set the experiment name to an experiment 
if len(sys.argv) > 3:
    mlflow.set_experiment(sys.argv[3])
else:
    mlflow.set_experiment("prediction-bike-sharing")

##############################
#    Features and Target
##############################
data_processed_path = PATH_PROCESSED_DATA+'bikes_processed.csv'

bike_processed = pd.read_csv(data_processed_path )

X = bike_processed[bike_processed.columns.difference(['cnt', 'dteday'])].values
y = bike_processed['cnt'].values

print(bike_processed.head())
##############################
#    Train Test Split
##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

X_train_final = pd.DataFrame(X_train)
X_train_final['target'] = y_train

X_test_final = pd.DataFrame(X_test)
X_test_final['target'] = y_test 

# Save train data
X_train_final.to_csv(model_version_path+'/train.csv',index=False, encoding='utf-8-sig')
# Save test data
X_test_final.to_csv(model_version_path+'/test.csv', index=False, encoding='utf-8-sig')    

    
##############################
#    Train Model
##############################
print('Input matrix size {}'.format(bike_processed.shape))
print('X_train matrix size {}'.format(X_train.shape))
print('X_test matrix size {}'.format(X_test.shape))

#X = X_train_final[X_train_final.columns.difference(['target'])].values
#y= X_train_final['target'].values

#regr = RandomForestRegressor(max_depth=2, n_estimators=100, random_state=seed)
#regr.fit(X, y)  

with mlflow.start_run():
    
    mlflow.set_tag('algorithm', algorithm_name)
    mlflow.set_tag('model_version_name', model_version_name)

    
## GridSearch 
    param_search = {'n_estimators': np.random.randint(150, size=10), 
                    'max_depth': [10, 15, 20, 25], 
                    'min_samples_split': [2, 3, 4]}


    RandomForestRegressor_model =  RandomForestRegressor(random_state = seed, 
                                                max_depth = models_params.get('max_depth'),
                                                n_estimators = models_params.get('n_estimators'))
    
    scorer = make_scorer(RMSLE_metric, greater_is_better=False)
    grid_search_model = GridSearchCV(RandomForestRegressor_model, param_search, cv=5, scoring=scorer)
    grid_search_model.fit(X_train, y_train)

    print("GridSearch Finish! Best Params and Best Score: \n")
    print(grid_search_model.best_params_)
    print(grid_search_model.best_score_)
    
    # Use the best model
    models_params['max_depth'] = grid_search_model.best_params_['max_depth']
    models_params['n_estimators'] = grid_search_model.best_params_['n_estimators']
    models_params['min_samples_split'] = grid_search_model.best_params_['min_samples_split']
    
## Model Train   
    RandomForestRegressor_model = RandomForestRegressor(random_state = seed, 
                                                max_depth = models_params.get('max_depth'),
                                                n_estimators = models_params.get('n_estimators'),
                                                min_samples_split = models_params.get('min_samples_split'))
    
    RandomForestRegressor_model.fit(X_train, y_train)
    
    y_pred = RandomForestRegressor_model.predict(X_test)
   # y_pred = np.exp(y_pred)-1 # Non log
    
#    (MSLE, MSE, R2, MAE, RMSLE) = eval_metrics(y_test, y_pred)
    RMSLE = RMSLE_metric(y_test, y_pred)
    
    print("RandomForestRegressor:")
#    print("  MLSE: {}".format(MSLE))
#    print("  MSE: {}".format(MSE))
#    print("  MAE: {}".format(MAE))
#    print("  R2: {}".format(R2))
    print("  RMSLE: {}".format(RMSLE))
    
    # Log Params
    for param in models_params.keys():
        mlflow.log_param(param, models_params.get(param))
    
    # Log Metrics
#    mlflow.log_metric("MSLE", MSLE)
#    mlflow.log_metric("MSE", MSE)
#    mlflow.log_metric("MAE", MAE)
#    mlflow.log_metric("R2", R2)
    mlflow.log_metric('RMSLE', RMSLE)
    
    mlflow.sklearn.log_model(RandomForestRegressor_model, "model")

    print('Modelo treinado com sucesso!')

    ##############################
    #    Plots
    ##############################
    # Plot the residuals
    residuals = y_test-y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_test, residuals)
    ax.axhline(lw=2,color='black')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Residuals')
    ax.title.set_text('Residual Plot | RMSLE: {}'.format(np.sqrt(RMSLE)))
    plot_residuls = "{}/{}-{}.png".format(model_version_path, 'model-residuals', model_version_name)
    plt.savefig(plot_residuls)
    
    # Plot Model
    plt.scatter(y_train, RandomForestRegressor_model.predict(X_train), alpha=0.5, color = 'red')
    plt.scatter(y_test, RandomForestRegressor_model.predict(X_test), alpha=0.5, color = 'blue')
    plt.title('Random Forest Model')
    plot_model = "{}/{}-{}.png".format(model_version_path, 'model-plot', model_version_name)
    plt.savefig(plot_model )
    
    mlflow.log_artifact(plot_residuls)
    mlflow.log_artifact(plot_model)
    #plt.show()

    ##############################
    #    Save Model
    ##############################  
#    #with open(os.path.join(model_version_path, '{}.pkl'.format(model_version_name)), 'wb') as fd:
#    with open(os.path.join(model_version_path, '{}.pkl'.format(model_version_name)), 'wb') as fd:
#        pickle.dump(regr, fd)
#        
#    #print('Modelo salvo em: {}'.format(model_version_path, '{}.pkl'.format(model_version_name)))
#    print('Modelo salvo em: {}'.format(model_version_path, '/{}.pkl'.format(model_version_name)))
    ##############################
    #    Log Artifacts
    ##############################
    # Log Data Processed
    mlflow.log_artifact(local_path=data_processed_path)
    # Log all artifacts generated during the experiment
    mlflow.log_artifacts(local_dir=model_version_path)