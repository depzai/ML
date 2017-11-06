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
import pandas as pd
from pandas.plotting import scatter_matrix

housing = pd.read_csv("housing.csv")
print(housing.head())

