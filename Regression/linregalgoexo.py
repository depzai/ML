from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import quandl

quandl.ApiConfig.api_key = "U5zALDafBdzSVuVmipmJ"

style.use('fivethirtyeight')

df = quandl.get("WIKI/AMZN")

print(df.head(50))

df = df.tail(1000)


df = df.reset_index()
df = df[['Adj. Close']]
df.columns = ['Close']


xs = np.array(df.index)
ys = np.array(df.Close)

def gety(x_input):
	m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs)**2 - mean(xs**2))
	b = mean(ys) - m * mean(xs)
	res = m * x_input + b
	return res

avgy = gety(xs)
meany = [mean(ys) for i in range(0,1000)]
# for i in range(0,199):
#  	meany.append(mean(ys))
# meany = np.array(meany)

plt.scatter(xs,ys, s=1)
plt.plot(xs, avgy, color = 'g', linewidth=1)
plt.plot(xs, meany, color = 'r', linewidth=1)
plt.show()

def squared_error(a_s, b_s):
	se = sum((a_s - b_s)**2)
	return se

print(squared_error(ys, avgy))
print(squared_error(ys, meany))

r_squared = 1 - squared_error(ys, avgy)/ squared_error(ys, meany)

print(r_squared)

