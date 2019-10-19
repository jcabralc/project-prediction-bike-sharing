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

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
#import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

#mlflow.delete_experiment('0')
#mlflow.delete_run('a913b1f95c9f4cf4aeef8f53855c889f')

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

algorithm_name = 'LGBMRegressor'
###############################
##  Function Evaluation
###############################
def eval_metrics(actual, pred):
    MSLE = mean_squared_log_error(actual, pred) 
    MSE = mean_squared_error(actual, pred) 
    R2 = r2_score(actual, pred)  
    MAE = mean_absolute_error(actual, pred)
    return MSLE, MSE, R2, MAE

###############################
##    Model Parameters
###############################
#max_depth = float(sys.argv[1]) # if len(sys.argv) > 1 else 0.5
#n_estimators = int(sys.argv[2]) # if len(sys.argv) > 2 else 0.5

models_params = { 
        'boosting_type':'gbdt', 
        'class_weight':None,
        'colsample_bytree':0.6746393485503049, 
        'importance_type':'split',
        'learning_rate':0.03158974434726661, 
        'max_bin':55, 
        'max_depth':-1,
        'min_child_samples':159, 
        'min_child_weight':0.001, 
        'min_split_gain':0.0,
        'n_estimators':1458, 
        'n_jobs':-1, 
        'num_leaves':196, 
        'objective':None,
        'random_state':seed, 
        'reg_alpha':0.23417614793823338,
        'reg_lambda':0.33890027779706655, 
        'silent':True,
        'subsample':0.5712459474269626, 
        'subsample_for_bin':200000,
        'subsample_freq':1}
              
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
                   
    LGBMRegressor_model = LGBMRegressor(boosting_type = models_params.get('boosting_type'), 
                                        class_weight = models_params.get('class_weight'),
                                        colsample_bytree = models_params.get('colsample_bytree'), 
                                        importance_type = models_params.get('importance_type'),
                                        learning_rate = models_params.get('learning_rate'), 
                                        max_bin=models_params.get('max_bin'), 
                                        max_depth=models_params.get('max_depth'),
                                        min_child_samples=models_params.get('min_child_samples'), 
                                        min_child_weight=models_params.get('min_child_weight'), 
                                        min_split_gain=models_params.get('min_split_gain'),
                                        n_estimators=models_params.get('n_estimators'), 
                                        n_jobs=models_params.get('n_jobs'), 
                                        num_leaves=models_params.get('num_leaves'), 
                                        objective = models_params.get('objective'),
                                        random_state = seed, 
                                        reg_alpha = models_params.get('reg_alpha'),
                                        reg_lambda = models_params.get('reg_lambda'), 
                                        silent = models_params.get('silent'),
                                        subsample = models_params.get('subsample'), 
                                        subsample_for_bin = models_params.get('subsample_for_bin'),
                                        subsample_freq = models_params.get('subsample_freq'))
    LGBMRegressor_model.fit(X_train, y_train)
    
    y_pred = LGBMRegressor_model.predict(X_test) # log predictions
    y_pred = np.exp(y_pred)-1 # Non log
    
    (MSLE, MSE, R2, MAE) = eval_metrics(y_test, y_pred)
    
    print("LGBMRegressor:")
    print("  MLSE: {}".format(MSLE))
    print("  MSE: {}".format(MSE))
    print("  MAE: {}".format(MAE))
    print("  R2: {}".format(R2))
    
    # Log Params
#    mlflow.log_param("boosting_type", 'gbdt')
#    mlflow.log_param("colsample_bytree", '0.6746393485503049')
#    mlflow.log_param("importance_type", 'split')
#    mlflow.log_param("learning_rate", '0.03158974434726661')
#    mlflow.log_param("max_bin", '55')
#    mlflow.log_param("max_depth", '1')
#    mlflow.log_param("min_child_samples", '159')
#    mlflow.log_param("min_child_weight", '0.001')
#    mlflow.log_param("min_split_gain", '0.0')
#    mlflow.log_param("n_estimators", '1458')
#    mlflow.log_param("n_jobs", '1')
#    mlflow.log_param("num_leaves", '196')
#    mlflow.log_param("objective", 'None')
#    mlflow.log_param("random_state", seed)
#    mlflow.log_param("reg_alpha", '0.23417614793823338')
#    mlflow.log_param("reg_lambda", '0.33890027779706655')
#    mlflow.log_param("silent", 'False')
#    mlflow.log_param("subsample", '0.5712459474269626')
#    mlflow.log_param("subsample_for_bin", '200000')
#    mlflow.log_param("subsample_freq", '1')
    for param in models_params.keys():
        mlflow.log_param(param, models_params.get(param))
    
    # Log Metrics
    mlflow.log_metric("MSLE", MSLE)
    mlflow.log_metric("MSE", MSE)
    mlflow.log_metric("MAE", MAE)
    mlflow.log_metric("R2", R2)
    
    mlflow.sklearn.log_model(LGBMRegressor_model, "model")

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
    ax.title.set_text('Residual Plot | Root Squared Mean Log Error: {}'.format(np.sqrt(MSLE)))
    plot_residuls = "{}/{}-{}.png".format(model_version_path, 'model-residuals', model_version_name)
    plt.savefig(plot_residuls)
    mlflow.log_artifact(plot_residuls)
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