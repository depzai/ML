
from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt 
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

# plot1 = (1,3)
# plot2 = (2,5)

# euclidian_distance = sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)

# print(euclidian_distance)

#create a dictionary with 2 classes:
dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_feature = [5,7]

#plotting loop: i as classes (k and r), ii as their attribute lists
for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1], s = 100, color = i)
plt.scatter(new_feature[0], new_feature[1])
plt.show()

#define k nearest neigbor algoP: which one is the nearest datapoint: have to calculate all
def k_nearest_neighbors(data, predict, k=3):
	if len(data) >=k:
		warnings.warn('K is set to a value less than total voting group')

	distances = []
	for group in data:
		for features in data[group]:
			#NOt fast enough so use numpy formula instead
			#euclidian_distance = sqrt((feature[0]-predict[0])**2+(feature[1]-predict[1])**2)
			euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidian_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

result = k_nearest_neighbors(dataset, new_feature, k=3)
print(result)








