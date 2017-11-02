import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.utils import shuffle
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.dropna(inplace = True)
df.replace('?', -999999, inplace=True)
df.drop(['id'], 1, inplace=True)

df = shuffle(df)
df = df.reset_index(drop= True)

labels = pd.DataFrame(df['class'])

df.drop(['class'], 1, inplace=True)

print(df.head())
print(labels)

print(df.shape)

X_train = np.array(df[:40])
y_train = np.array(labels['class'][:40])

X_test = np.array(df[-40:-1])
y_test = np.array(labels['class'][-40:-1])

clf = svm.SVC()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
































