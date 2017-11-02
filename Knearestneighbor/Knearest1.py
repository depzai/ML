
# in dataset: numbers 1 to 10, 
# classes: 2 is benign, 4 is malignant

import pandas as pd 
import numpy as np 
from sklearn import preprocessing, model_selection 
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#replace '?' in dataset

df.replace('?', -99999, inplace = True)
# print(df)
df.drop(['id'], 1, inplace=True)

# X is features, y is labels

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


#use 20% of datat to train and to test
X_train, X_test, y_train, y_test, = model_selection.train_test_split(X,y,test_size = 0.2)

clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
print(example_measures)
#NEED 2d array instead of 1d array so reshape:
example_measures = example_measures.reshape(1,-1)
print(example_measures)
prediction = clf.predict(example_measures)

print(prediction)