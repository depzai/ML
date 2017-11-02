import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma =0.001, C=100)

# load x and y with data from digits.data: leave the last 10 for testing
x,y = digits.data[:-10], digits.target[:-10]

print(len(digits.data))

#fit data to learn: fits a line through values
clf.fit(x,y)
# predict the element in -2 position, outside of training area
print('Prediction:', clf.predict(digits.data[-2]))

# shows what the element we are trying to interpret looks like
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")

plt.show()



