import numpy as np 
import pandas as pd 
from pandas.plotting import scatter_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import sys


data = pd.read_csv('data.csv')

print(data.head())

#CORRELATIONS - NOT HELPFUL HERE SINCE TARGET IS 1 and 0:
# correlations = data[['speechiness', 'liveness', 'loudness', 'danceability', 'acousticness', 'target']]
# corr_matrix = correlations.corr()

# print(corr_matrix)
# print(scatter_matrix(correlations, alpha=0.2, figsize=(6, 6), diagonal='kde', color = 'b'))
# plt.show()

#HISTOGRAM OF VARIABLE with Target 1 (like music)

danceability_1 = data['danceability'][data['target'] == 1]

# print(danceability_1.head())

# plt.hist(danceability_1, 50)
# plt.xlabel('danceability')
# plt.ylabel('Positives')
# plt.title('Pos resp to danceability')
# plt.show()

X = data[['speechiness', 'liveness', 'energy', 'instrumentalness', 'tempo', 'valence', 'loudness', 'danceability', 'acousticness']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

y_test_array = np.array(y_test)

results = pd.DataFrame()
results['pred'] = pred
results['actual'] = y_test_array

results['score'] = ''

for i in range(len(pred)):
	if results['pred'][i] == results['actual'][i]:
		results['score'][i] = 1
	else:
		results['score'][i] = 0


accuracy = results['score'].sum()/len(results['score'])

print("the accuracy is {}".format("{:.2%}".format(accuracy)))



