import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
import random

x_values = []
y_values = []
z_values = []

for i in range(0,20):
	val = i + random.randrange(-10,10,2)
	x_values.append(val)

for j in range(0,20):
	val = j + random.randrange(-10,10,2)
	y_values.append(val)

for k in range(0,20):
	val = k + random.randrange(-10,10,2)
	z_values.append(val)

X = np.array([x_values[0], y_values[0], z_values[0]])
for i in range(1,len(x_values)):
	X = np.vstack([X, [x_values[i], y_values[i], z_values[i]]])

print(X)


ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x_values,y_values, z_values, color = 'g')
plt.scatter(cluster_centers[0][0], cluster_centers[0][1], cluster_centers[0][2], color = 'r')
plt.scatter(cluster_centers[1][0], cluster_centers[1][1], cluster_centers[1][1], color = 'r')
plt.show()





