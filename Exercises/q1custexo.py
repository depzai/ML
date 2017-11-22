import numpy as np 
import pandas as pd 
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

df = pd.read_csv('q1cust.csv')

# print(df.head(50))



df.fillna(value = 0, inplace = True)

cancels = df[df['date_cancel'] != 0]
cancels = cancels[cancels['revenue'] < 100]
cancels = cancels[cancels['revenue'] > 0]

print(cancels.head())

dico = {}

cancel_dates = cancels.date_cancel.unique()

for i in range(len(cancel_dates)):
	dico[cancel_dates[i]] = (i+1)*10

print(dico)

date_cancel_list = list(cancels['date_cancel'])

for i in range(len(date_cancel_list)):
	date_cancel_list[i] = dico[date_cancel_list[i]]

cancels['date_cancel'] = date_cancel_list

print(cancels.head())

X1 = list(cancels['date_cancel'])
X2 = list(cancels['revenue'])

plt.scatter(X2, X1, color = 'b', marker = 'o', s = 1)

plt.show()
