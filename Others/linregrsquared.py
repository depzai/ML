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


#define squared error: variables ys_orig and ys_line are numpy arrays
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig] #will be an array with only the mean value of all ys'
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    print("y_mean_line", y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


print(b)

regression_line = [m*x + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)

print(r_squared)
predict_x = 8

predict_y = m * predict_x + b

print(predict_y)

print(regression_line)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(regression_line)
plt.show()

