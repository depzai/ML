from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)



# best fit slope
def best_fit_slope(xs,ys):
	m = (mean(xs) * mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
	return m

m = best_fit_slope(xs, ys)

print(m)

def best_fit_line(xs,ys):
	b = mean(ys) - m * mean(xs)
	return b

b = best_fit_line(xs,ys)

print(b)

regression_line = [m*x + b for x in xs]

predict_x = 8

predict_y = m * predict_x + b

print(predict_y)

print(regression_line)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(regression_line)
plt.show()

# how good of a fit is this best fit line?





