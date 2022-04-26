# -*- coding: utf-8 -*-
"""
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



#loading dataset

os.chdir("C:/Users/Lucia/Desktop/TESIS/Datos")
files = os.listdir("C:/Users/Lucia/Desktop/TESIS/Datos")

df = pd.DataFrame()
for file in files:
    if file.endswith('.xlsx'):
        df = df.append(pd.read_excel(file))
        

# Adding datetime and changing month number into month str

df['date'] = pd.to_datetime( df[['year', 'month', 'day']])

#splitting stations


stations_id = df['station'].value_counts().index.tolist() #to know how many stations are

centenario = df.loc[df["station"] == stations_id[0]]
cordoba = df.loc[df["station"] == stations_id[1]]
laboca = df.loc[df["station"] == stations_id[2]]   

pollutants = ['NO2', 'PM10', 'CO']
stations = [centenario, cordoba, laboca]
df_list = []
       

# Count NaN, add date index, day of the week, decompose WD


for station in stations:
    for pollutant in pollutants:
        nan = station[pollutant].isnull().sum()
        percentage = nan/len(station) * 100
        station_id =station['station'][0]
        print(f'{pollutant} in {station_id} has {nan} NaN - {percentage} % of data missing')
        df_id = station[['date','year','month','day','hour', 'station','AT','RH','SC','WD','WS', pollutant]]
        df_id = df_id.drop(df_id.index[0:48])
        df_id = df_id.drop(df_id.index[90552:90576])
        df_id["indice"] = np.arange(90552)
        x = np.array([[1,2,3,4,5,6,7]])
        y = np.repeat(x, 24)
        df_id ["weekday"] = np.tile(y, 539)
        df_id ["u"] = df_id["WS"] * (np.sin( (df_id["WD"] * np.pi / 180) + np.pi))  
        df_id ["v"] = df_id["WS"] * (np.cos( (df_id["WD"] * np.pi / 180) + np.pi)) 
        df_list.append(df_id)

# Write in xlsx

writer = pd.ExcelWriter('data_prep_pollution.xlsx')
for i in np.arange(9):
    df_list[i].to_excel(writer, sheet_name=f'sheetName_{i}')
        
