# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:08:54 2021

@author: Bhawna
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

data = pd.read_csv('C:/Users/Bhawna/.spyder-py3/minute_weather.csv')
print(data.head())
print(data.shape)


sampled_df = data[(data['rowID'] % 10) == 0]
print('sampled data..',sampled_df.shape)

sampled_df.describe().transpose()

sampled_df[sampled_df['rain_accumulation'] == 0].shape

sampled_df[sampled_df['rain_duration'] == 0].shape

del sampled_df['rain_accumulation']
del sampled_df['rain_duration']

rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]

rows_before - rows_after
print('sampled_df---columns',sampled_df.columns)

features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 
        'max_wind_speed','relative_humidity']

select_df = sampled_df[features]

print('select_df--columns',select_df.columns)
print('....... select df data.....',select_df.head())

X = StandardScaler().fit_transform(select_df)
#print(' standard scale data..',X.head())

kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)

centers = model.cluster_centers_
print('...centers are...',centers)

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
    
P = pd_centers(features, centers)
print('*** P ***',P)

print('........Dry days Plot.....')
parallel_plot(P[P['relative_humidity'] < -0.5])

print('...Warm days plot....')
parallel_plot(P[P['air_temp'] > 0.5])

print('.....cool days plot....')
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])