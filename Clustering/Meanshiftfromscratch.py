import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift

X = np.array([[1, 2],
              [1.5, 1.8],
              [4, 10 ],
              [5, 10 ],
              [4.5, 11 ],
              [9, 10 ],              
              [8, 10],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

# plt.scatter(X[:,0], X[:,1], s=150, color = 'b')
# plt.show()

colors = 10*["g","r","c","b","k"]

# first step: every data point is a cluster center


# using sklearn
clf = MeanShift(bandwidth = 2)
clf.fit(X)

labels = clf.labels_
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

centroids = clf.cluster_centers_
print(centroids)

plt.scatter(X[:,0], X[:,1], s=150)

for c in range(len(centroids)):
     plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()