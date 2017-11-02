import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
import sklearn
from sklearn.cluster import KMeans

x_values = []
y_values = []

for i in range(0,20):
	val = i + random.randrange(-10,10,2)
	x_values.append(val)

for j in range(0,20):
	val = j + random.randrange(-10,10,2)
	y_values.append(val)

X = np.array([x_values[0], y_values[0]])
for i in range(1,len(x_values)):
	X = np.vstack([X, [x_values[i], y_values[i]]])

print(X)

clf = KMeans(n_clusters = 2)
clf.fit(X)

print(clf.cluster_centers_)

centroid1 = clf.cluster_centers_[0]
centroid2 = clf.cluster_centers_[1]

plt.scatter(x_values, y_values, color='g')
plt.scatter(centroid1[0], centroid1[1],marker = "x", s=150, color = 'b')
plt.scatter(centroid2[0], centroid2[1],marker = "x", s=150, color = 'b')

plt.show()