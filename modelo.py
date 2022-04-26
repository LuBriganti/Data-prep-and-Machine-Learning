# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:09:19 2022

@author: Lucia
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:06:49 2022

@author: Lucia
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:50:10 2022

@author: Lucia
"""


import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
import random
import statistics as stats

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split 
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#loading dataset

os.chdir("C:/Users/Lucia/Desktop/TESIS/Datos")
files = os.listdir("C:/Users/Lucia/Desktop/TESIS/Datos")

dfname=pd.ExcelFile('data_prep_pollution.xlsx')
print(dfname.sheet_names)
df=pd.read_excel('data_prep_pollution.xlsx')

df_list = []


for items in dfname.sheet_names[1:]:
    dfnew=pd.read_excel('data_prep_pollution.xlsx',sheet_name=items)
    df_list.append(dfnew)


## ML setting and implementation


loss = ['ls', 'lad', 'huber']
n_estimators =  [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
LEARNING_RATE = 0.05
subsample = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
min_samples_leaf = np.arange(start = 0, stop =31, step = 1)
max_depth = [2, 3, 5, 10, 15, 20, 25]
min_samples_split = [2, 4, 6, 10]
max_features = ['auto', 'sqrt', 'log2', None]
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features,
                       "subsample" : subsample}


for i in np.arange(9):

        data = df_list[i]
        data = data.dropna()
        station = data['station'][0]
        pollutant = list(data.columns.values)[-1]
        subdata = data.set_index("date")
        subdata = subdata.drop(['station'], axis = 1)


        subdata = subdata.loc['2017-01-01':'2019-12-31']
        subdata_random = subdata.sample(frac = 1)
                
        X_train, X_test = train_test_split(subdata_random, train_size=0.9, test_size=0.1)

        y_train = X_train.pop(pollutant)
        y_test = X_test.pop(pollutant)
        media_test = stats.mean(y_test)


        model = GradientBoostingRegressor(random_state = 42)


        random_cv = RandomizedSearchCV(estimator=model,
                                       param_distributions=hyperparameter_grid,
                                       cv=5, n_iter=50,
                                       scoring = 'neg_root_mean_squared_error',
                                       n_jobs = -1, verbose = 1,
                                       return_train_score = True,
                                       random_state=42)
                                
        random_cv.fit(X_train, y_train)
                                
        random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)
        random_results.to_excel(f'hiperparm & metrics {station} {pollutant}',engine='xlsxwriter')
                                
                                
        final_model = random_cv.best_estimator_
        final_model.fit(X_train, y_train)
        final_pred = final_model.predict(X_test)


        MAE = mean_absolute_error(final_pred, y_test)
        rMSE = mean_squared_error(final_pred, y_test,squared = False )
        nRMSE = rMSE / media_test
        r2 = r2_score(final_pred, y_test)
        metricos_ = [[MAE, rMSE, nRMSE ,r2]]
        dfmetricos = pd.DataFrame(metricos_ , columns = ["MAE", "rMSE","nRMSE","r2"])
        dfmetricos.to_excel(f' hiperparam & metrics {station} {pollutant}', engine='xlsxwriter', sheet_name = 'sheet2')
