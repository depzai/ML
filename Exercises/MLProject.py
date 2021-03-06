import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
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
# Create an income category column with 5 categories:
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
#Any income over 5 is 5.0
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
print(housing.head())
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]
print(housing['income_cat'].value_counts() / len(housing))
#then drop income cat column from test sets:
strat_train_set.drop('income_cat', 1, inplace=True)
strat_test_set.drop('income_cat', 1, inplace=True)

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
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending = False))

#CORRELATION USING PANDAS SCATTER MATRIX
#Looking for correlations for long, lat, pop, mi, mhv

# housing = housing[['longitude', 'latitude', 'population', 'median_income', 'median_house_value']]
# print(scatter_matrix(housing, alpha=0.2, figsize=(6, 6), diagonal='kde', color = 'b'))
# plt.show()

#Prepare data for ML - training set
y_train = strat_train_set['median_house_value'].copy()
X_train = strat_train_set.drop('median_house_value', 1, inplace=True)


#Data Cleaning
#IMPUTER from sklearn replaces missing values with median of attribute

imputer = Imputer(strategy = 'median')

#ocean proximity doesn't have values, so we'll drop it for this
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)

housing_num = housing.drop('ocean_proximity', 1)

print(housing_num.head())

imputer.fit(housing_num)
X = imputer.transform(housing_num)

# #Change ocean proximity to numbers:
# encoder = LabelEncoder()
# housing_cat = housing['ocean_proximity']
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# issue is that in this instance the numbers aren't in order
# can be fixed with OneHotEncoder or with custom function

#Feature scaling: in ML, attritbutes have to have a similar scale
# DAta should be resized to have similar scale

#SKLEARN only works with np arrays, not dataframes

# Prepare data using Pipelines:
housing_prepared = pd.read_csv('housing_prepared.csv')
housing_labels = pd.read_csv('housing_labels.csv')

housing_prepared = np.array(housing_prepared)
housing_labels = np.array(housing_labels)

print(housing_prepared)


# LInear Reg:
X = housing_prepared
y = housing_labels

clf = LinearRegression()

clf.fit(X,y)

print(len(X), len(y))


print(X)














