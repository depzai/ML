import quandl, math, datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

quandl.ApiConfig.api_key = "U5zALDafBdzSVuVmipmJ"

style.use('ggplot')

df = quandl.get("WIKI/AMZN")

df = df.tail(500)

# print(df.shape)

df = df.reset_index()
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.dropna()
# print(df.head(50))

#populate new fields with past values
for i in range(500,550):
	df.loc[i] = df.loc[i-100]

df['Labels'] = df.Close

# print(df.tail(60))

clf = LinearRegression()

y = np.array(df.Labels)
df = df[['Open', 'High', 'Low', 'Volume']]

X = np.array(df)

# print(X)
# print(y)

X_train = X[0:50]
X_test = X[51:101]
y_train = y[0:50]
y_test = y[51:101]

clf.fit(X_train,y_train)

#Pickling to save the trained classifier
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf,f)
pickle_in = open('linearregression.pickle', 'rb')
#clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

y[500:-1] = clf.predict(X[500:-1])

print(accuracy)

dates = np.array(df.index)

plt.plot(dates,y)
plt.show()