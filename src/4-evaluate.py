# -*- coding: utf-8 -*-
"""
@author: Jessica Cabral
"""
############################################################################
#       Evaluate Script
############################################################################

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics

try:
    import cPickle as pickle
except ImportError:
    import pickle

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython evaluate.py models-folder output\n')
    sys.exit(1)

model_file = os.path.join(sys.argv[1], 'model.pkl') #' sys.argv[2])
matrix_file = os.path.join(sys.argv[1], 'test.tsv')
#metrics_file = os.path.join(sys.argv[1], sys.argv[2])
metrics_file = sys.argv[2]

print(model_file)
print(matrix_file)
print(metrics_file)

##############################
#    Import test data and model
##############################
with open(model_file , 'rb') as fd:
    model = pickle.load(fd)

with open(matrix_file, 'rb') as fd:
    matrix = pd.read_csv(fd, sep='\t')

##############################
#    Make de predictions
##############################
X = matrix[matrix.columns.difference(['target'])].values
y_true = matrix['target'].values



##############################
#    Metrics
##############################
MSLE = mean_squared_log_error(y_true, y_pred) 
MSE = mean_squared_error(y_true, y_pred) 
R2 = r2_score(y_true, y_pred)  
MAE = mean_absolute_error(y_true, y_pred)

# Plot the residuals
residuals = y_true-y_pred
fig, ax = plt.subplots()
ax.scatter(y_true, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text('Residual Plot | Root Squared Mean Log Error: {}'.format(np.sqrt(MSLE)))
plt.savefig(os.path.join(sys.argv[1], 'model-residuals.png'))
plt.show()

##############################
#    Save Metrics file DVC
##############################
with open(metrics_file, 'w') as fd:
    fd.write('MSLE: {:4f}\n'.format(np.sqrt(MSLE)))
    fd.write('MSE: {:4f}\n'.format(MSE))
    fd.write('R2: {:4f}\n'.format(R2))
    fd.write('MAE: {:4f}\n'.format(MAE))


print('Metrics file saved!')