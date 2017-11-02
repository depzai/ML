
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

# Class: last number is class #: 1 or 2
class1 = np.array([[1,2,1],[1,3,1],[2,3,1],[3,4,1],[2,4,1],[1,4,1]])
class2 = np.array([[5,6,2],[5,7,2],[6,7,2],[7,8,2],[6,8,2],[5,8,2]])
new_element = [3,5]

class1_x = class1[:,0]
class1_y = class1[:,1]

class2_x = class2[:,0]
class2_y = class2[:,1]

plt.scatter(class1_x, class1_y, s = 20, color = 'g')
plt.scatter(class2_x, class2_y, s = 20, color = 'b')
plt.scatter(new_element[0],new_element[1], s = 10, color = 'r')
plt.show()

def euclidian_distance(a,b):
	calc = sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
	return calc

#better to use dictionary instead
results = {'class1':[], 'class2':[]}

for i in range(0,len(class1)):
	p = euclidian_distance(new_element,class1[i])
	results['class1'].append(p)

for i in range(0,len(class2)):
	p = euclidian_distance(new_element,class2[i])
	results['class2'].append(p)

results['class1'].sort()
results['class2'].sort()

print(results)
df1 = pd.DataFrame(results['class1'], columns=['distance'])
df1['label'] = 1
df2 = pd.DataFrame(results['class1'], columns=['distance'])
df2['label'] = 2
df = pd.merge(df1, df2, on=['distance','label'], how='outer')
df = df.sort_values(by = 'distance', ascending = 1)
print(df)

k = 5

df = df[:k]
print('Nearest class is (#instances)', Counter(df.label).most_common())



