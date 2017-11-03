import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from pandas.plotting import scatter_matrix

housing = pd.read_csv("housing.csv")
print(housing.head())


#Info about dataset:
# housing.info()
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())

# Histogram for whole dataset:
# housing.hist(bins = 50, figsize =(20,15))
# plt.show()

#Create a test set: 20% of data
# housing2 = housing.sample(frac = 1)
# test_set = housing2[-4128:-1]
# train_set = housing2[:-4128]

# SAmpling using sklearn
# using 42 as random state generates the same sample every time. 42 is random, could be any number
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

#SAMPLING USING PARAMETERS TO GET A SAMPLE THAT IS REPRESENTATIVE of dataset
# # Create an income category column with 5 categories:
# housing['income_cat'] = np.ceil(housing['median_income']/1.5)
# #Any income over 5 is 5.0
# housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
# print(housing.head())
# split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
# for train_index, test_index in split.split(housing, housing['income_cat']):
# 	strat_train_set = housing.loc[train_index]
# 	strat_test_set = housing.loc[test_index]
# print(housing['income_cat'].value_counts() / len(housing))
# #then drop income cat column from test sets:
# strat_train_set.drop('income_cat', 1, inplace=True)
# strat_test_set.drop('income_cat', 1, inplace=True)

#Explore Data
# plt.scatter(housing['longitude'], housing['latitude'],s=10, color = 'b')
# plt.show()
#with density of data point using alpha = 0.1:
# plt.scatter(housing['longitude'], housing['latitude'],s=10, color = 'b', alpha = 0.1)
# plt.show()

# housing.plot(kind= 'scatter',x='longitude', y='latitude', alpha=0.4,
# 	s=housing['population']/100, label = 'population', c='median_house_value',
# 	cmap=plt.get_cmap('jet'), colorbar=True)
# plt.show()

# CORRELATION WITH MEDIAN HOUSE VALUE
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending = False))

#CORRELATION USING PANDAS SCATTER MATRIX
#Looking for correlations for long, lat, pop, mi, mhv

housing = housing[['longitude', 'latitude', 'population', 'median_income', 'median_house_value']]
print(scatter_matrix(housing, alpha=0.2, figsize=(6, 6), diagonal='kde', color = 'b'))
plt.show()


















