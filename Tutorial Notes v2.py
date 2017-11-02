#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 03:08:45 2017

@author: bddupont
"""
>>> width = 20
>>> height = 5 * 9
>>> width * height
>>> BMI = width/height**2
>>> Type(BMI) -- type integer or float etc...
>>> 'spam eggs'  # single quotes
'spam eggs'
>>> 'doesn\'t'  # use \' to escape the single quote...
"doesn't"
>>> "doesn't"  # ...or use double quotes instead
"doesn't"
>>> '"Isn\'t," she said.'
'"Isn\'t," she said.'
>>> print('"Isn\'t," she said.')
"Isn't," she said.
>>> s = 'First line.\nSecond line.'  # \n means newline
>>> s  # without print(), \n is included in the output
'First line.\nSecond line.'
>>> print(s)  # with print(), \n produces a new line
First line.
Second line.
>>> # 3 times 'un', followed by 'ium'
>>> 3 * 'un' + 'ium'
>>> prefix = 'This value will be defined as'
>>> prefix + 'n'

#Character position
>>> word = 'Python'
>>> word[0]  # character in position 0
'P'
>>> word[5]  # character in position 5
'n'
>>> word[0:3]
'pyt'
>>> word[:2]   # character from the beginning to position 2 (excluded)
'Py'
>>> word[4:]   # characters from position 4 (included) to the end
'on'
>>> word[-2:]  # characters from the second-last (included) to the end
'on'
# LENGTH
>>> s = 'supercalifragilisticexpialidocious'
>>> len(s)
34

# input and embed values in text:
print('what is your age', end = ' ')
age = input()
print('what is your height', end = ' ')
height = input()
print(f"so your age is {age} and your height is {height}")

#LISTS
>>> squares = [1, 4, 9, 16, 25]
>>> squares
[1, 4, 9, 16, 25]
#Uses len and position in the same way
#A list can contain a list within a list
squares[0] --> 1
squares[1:5]
squares[:4]
squares[2:]
#Last element:
squares[-1]

# Change List
a = [1,3,5,7,10]
a[4] = 9 # to change 10 to 9
#OR
a[-1] = 9
# APPEND - add to a list
>>> a.append(11)
>>> a
[1, 3, 5, 7, 9, 11]

Print('First Item', [0])
b = ['qa', 'we']

c = [a, b] # list inside of listx[]

print(c[1][1])# in second listprint 2nd object

a.remove(3)# will remove that element 

list1 = ['qa,'we']
list1.remove('qa')
#or
del(a[1])
list1.insert(1,'as') # inserts inthat position 1 the following character 'as'

#Max:
Print(max(list1))


#Copy list:
x = [1,2,3]
y = x # now y and x refer to the same list
y = list(x) #copy x elements into y
y = x[:] # same thing



# Methods - funtions that apply to objects (lists, strings etc)
fam = ['liz', 1.73, 'jim',1.85,'bob',1.90]
fam.index("bob") # which position is bob at
fam.count(1.73) # how many times is 1.73 in the list
fam,replace("j","t")
fam.append('joe')


#Tuples:
NOt possible to change

pi_tuple = (93,1,4,5,9)
new_tuple = list(pi_tuple) # cnvert tuple into list
#len, max, min for tuples


# Dictionaries: item and its definition (value)
super_villains = {'Fiddler' : 'Isaac bowin', 'Captain Cold' : 'Leonard Snart'}
print(super_villains['Captain Cold'])
print(super_villains.get('Fiddler'))
print(super_villains.values())

# COnditionals
age = 21
if age > 16:
    print('old enogh to drive')
else :
    print('don't drive')

if age>=21:
    print('you can drive a tractor')
elif age >= 16:
    print('dirve a car')
else :
print('dont drive')


if((age >1) and (age <=18) ):
print('party')
elif(age == 21) or (age >= 65):
print('party')
elif not (age == 30):
print('no party')
else:
print('party, yeah')

# loops:
# make system do something starting a 0 and 10 times
for x in range (0,10):
    print(x,' ', end = "")

Print('n\') # print new line

grocery = ['juice', 'tomatoes', 'potatoes']

for y in grocery:
    print(y)
    

# Wild Loops
import random
random_num = random.randrange(0,100)

#Print random numbers until you get to 15:
while(random_num != 15):
    print(random_num)
    random_num = random.randrange(0,100) #reset random number after print

#Using iterator to get even numbers until 11:
i = 0;
while (i<=20):
    if(i%2 == 0):
        print(i)
    elif(i ==11):
        break
    else:
        i = i+1
        continue
    i = i + 1


#Functions

# get help on function: help(max)

def addNumber(fNum, lNum):
    sumNum = fNum + lNum
    return sumNum
print(addNumer(1,4)
#or string = sumNum(1,4)
#What happens in functions is only true within the function unless it is returned

import sys
print('What is your name')
name = sys.stdin.readline()
print('Hello', name)

#Print characters of a string

string1 = "I'll get you"
print(string1[0:4])
print(string1[:-5])
print(string1[0:4] + " be there")

print("%c is my %s letter ad my number %d is %.5f" %('X','favorite', 1, .14))

print(string1.replace("you","me"))


#External File
test_file = open("test.txt","wb") #open file and want to be able to write into it

print(test_file.mode) #show file and mode(in this case wb)
print(test_finle.name)
#write text:
test_file.write(bytes("Write me to the file\n",'UTF-8'))
test_file.close

#read info from the file:
test_file = open("etst.txt", "r+")
text_in_file = test_file.read()
print(text_in_file)

#delete file:
import os
os.remove("test.txt")




#OBJECTS. Objects have attributes (colors, heights etc). Attributes are variables in classes, abilities wil be functions
# CLASSES AND OBJECTS -------------
# The concept of OOP allows us to model real world things using code
# Every object has attributes (color, height, weight) which are object variables
# Every object has abilities (walk, talk, eat) which are object functions
 
class Animal:
    # None signifies the lack of a value
    # You can make a variable private by starting it with __
    __name = None
    __height = None
    __weight = None
    __sound = None
 
    # The constructor is called to set up or initialize an object
    # self allows an object to refer to itself inside of the class
    def __init__(self, name, height, weight, sound):
        self.__name = name
        self.__height = height
        self.__weight = weight
        self.__sound = sound
 
    def set_name(self, name):
        self.__name = name
 
    def set_height(self, height):
        self.__height = height
 
    def set_weight(self, height):
        self.__height = height
 
    def set_sound(self, sound):
        self.__sound = sound
 
    def get_name(self):
        return self.__name
 
    def get_height(self):
        return str(self.__height)
 
    def get_weight(self):
        return str(self.__weight)
 
    def get_sound(self):
        return self.__sound
 
    def get_type(self):
        print("Animal")
 
    def toString(self):
        return "{} is {} cm tall and {} kilograms and says {}".format(self.__name, self.__height, self.__weight, self.__sound)
 
# How to create a Animal object
cat = Animal('Whiskers', 33, 10, 'Meow')
 
print(cat.toString())


# create Dog class with cat attributes:
class Dog(Animal):
    __owner = None
 








>>> # Fibonacci series wih multiple assignment:
... # the sum of two elements defines the next
... a, b = 0, 1
>>> while b < 10:
...     print(b)
...     a, b = b, a+b
...
1
1
2
3
5
8

# %: for abs numbers only a%b means a-abs(a/b)

#%, ** is exp, ???

quote = "\"Always remember"
# MUlti line:
quote_2='''just
like everyone else'''

# Combine # strings:
combined = quote + quote_2    
print(combined)



# range:
>>> for i in range(2,5):
...     print(i)
2
3
4
5

# FOR
# Return prime numbers btw 10 and 15:
primes = [11,13]
for a in range(10,20):
    if a in primes:
        print (a,'prime')
    else:
        print (a,'not prime')
        
>>> a = ['Mary', 'had', 'a', 'little', 'lamb']
>>> for i in range(len(a)):
...     print(i, a[i])
...
0 Mary
1 had
2 a
3 little
4 lamb

# break: when clause in for statement become false
for letter in 'Python':
    if letter == 'h':
        print(letter,'foundit')
        break
    else:
        print(letter,'not h')
#vs.:
 for letter in 'Python':
    if letter == 'h':
        print(letter,'foundit')
    else:
        print(letter,'not h')   



#NUMPY
#numpy array: perfom calculations over entire arrays
# numpy performs calcs to arrays as if they are single values
# ONy one type in array for calcs

# height is available as a regular list

# Import numpy
import numpy as np

# Create a numpy array from height: np_height
np_height = np.array(height)

# Print out np_height
print(np_height)

# Convert np_height to m: np_height_m
np_height_m = np_height * 0.0254

# Print np_height_m
print(np_height_m)

# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])

# Create 2d array: table with 4 rows, 2 columns:
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball

np_baseball = np.array(baseball)
# Print out the type of np_baseball
print(np_baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)


# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])

# baseball is available as a regular list of lists
# updated is available as 2D numpy array

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)


# CSV Module
import csv

with open('example.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')
    for row in readCSV:
        print(row)






#read csv    
import pandas as pd
import numpy as np
import matplotlib as plt    
df = pd.read_csv("/Users/bddupont/Desktop/Docs/LTV - TM1/ALC for LTV - 0516.csv")                
df.head(10)

df[df.metric == "Customers"]

df.loc[df['metric'] == "Customers"]
df.loc[(df['metric'] == "Customers") & (df['start_date'] == "Apr-2017")]
df.loc[(df['metric'].isin(["Customers","Revenue"])) & (df['start_date'] == "Apr-2017")]

#Sumif:
df.loc[df['metric'] == "Customers", 'data'].sum()
df.loc[df['metric'] == "Customers", 'data'].count()
df.loc[(df['cohort_date'] == "Mar-11") & (df['country'] == "US"), 'revenue'].sum()


import csv

with open('example.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')
    
    offer_type = []
    metricname = []
    values = []
    
    
    for row in readCSV:
        offer = row[3]
        metric = row[4]
        value = row[5]
        
        offer_type.append(offer)
        metricname.append(metric)
        values.append(value)

del(offer_type[0])
del(metricname[0])
del(values[0])
        
print(offer_type)
print(metricname)
print(values)

OfferData = input('Which offer do you want to know the count of')
Offercount = offer_type.count(OfferData)

print('The count of', OfferData, 'is', Offercount)




#Create DataFrame:

import pandas as pd
import numpy as np    
tab = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
print(pd.DataFrame(data=tab[1:,1:],
                  index=tab[1:,0],
                  columns=tab[0,1:]))

Out:
     Col1 Col2
Row1    1    2
Row2    3    4

a = df.loc[df['metric'] == "Customers", 'data'].sum()

b = df.loc[df['metric'] == "Customers", 'data'].count()

tab = np.array([['','Col1','Col2'],
                ['Row1','sum',a],
                ['Row2','count',b]])

print(pd.DataFrame(data=tab[1:,1:],
                  index=tab[1:,0],
                  columns=tab[0,1:]))


#Pivot Data
df2 = df.loc[(df['metric'] == "Revenue") & (df['order_date'] == "Dec-2016") ]
df3 = df2.loc[(df['channels_listen'] ==("Sep-2016")) ]
df3.pivot(index='start_date', columns='sale_type_name', values='data')


# Pivot Table 
pd.pivot_table(df,index=["metric"]) # by default averages
pd.pivot_table(df,index=["metric", "order_date"])
pd.pivot_table(df,index=["metric","order_date"],values=["data"])

pd.pivot_table(df,index=["metric","order_date"],values=["data"],aggfunc=np.sum)

custlist = pd.pivot_table(df,index=["customer_key"], columns = ["yr_mth_name"],values=["sum"],aggfunc=np.sum)
pd.pivot_table(df,index=["metric","order_date"],values=["data"],aggfunc=[len]) # COUNT
#Pivot with Sum and Count;
pd.pivot_table(df,index=["metric","order_date"],values=["data"],aggfunc=[np.sum,len], fill_value = 0)# Fill value to replace n/a with 0

# Unique values
df.name.unique()

#Format
pd.options.display.float_format = '{:,.2f}'.format


# PANDAS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

web_stats = {'Day': [1,2,3,4,5,6],
             'Visitors': [43,53,34,45,64,34],
             'Bounce_Rate': [65,72,62,64,54,66]}

df = pd.DataFrame(web_stats)

print(df)
#print(df.head(3))
#print(df.tail(2))

# using Day as the index:

df2 = df.set_index('Day')

#print(df2)

#print(df.Visitors)
#print(df['Visitors'])

#OR for multiple columns

#print(df[['Bounce_Rate','Visitors']])

#Convert to list:
#print(df.Visitors.tolist())
#full_data = df.astype(float).values.tolist()
# Convert to Array:
print(np.array(df[['Bounce_Rate','Visitors']]))

# COnvert Array to DF:
    
df3 = pd.DataFrame(np.array(df[['Bounce_Rate','Visitors']]))

print(df3)


# PANDAS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


#IMport locally from same folder
#df = pd.read_csv('Housing.csv')
#print(df.head())

#Set index to Date into new csv file
#df.set_index('Date',inplace=True)
#df.to_csv('newcsv2.csv')

df = pd.read_csv('newcsv2.csv')
print(df.head())

#Take out first column
df = pd.read_csv('newcsv2.csv', index_col = 0)
print(df.head())

#rename columns
df.columns = ['Austin_HPI']
print(df.head())

df.to_html('example.html')

# Chnange value in row 0 (index) and column Revenue:
df.loc[custlist.index[0], 'Revenue'] = 99

import matplotlib.pyplot as plt
import sklearn
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')


#wildcard search "like"
df = file[file['device_category_desc'].str.contains('App')]


#MATPLOTLIB


# Support Vector Machines:
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma =0.001, C=100)

# load x and y with data from digits.data: leave the last 10 for testing
x,y = digits.data[:-10], digits.target[:-10]

#fit data to learn: fits a line through values
clf.fit(x,y)
# predict the element in -2 position, outside of training area
print('Prediction:', clf.predict(digits.data[-2]))

# shows what the element we are trying to interpret looks like
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")

plt.show()

# NAive Base:
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# X is the dataset with tenure (in x) and RPM (in Y)
X = np.array([[0, 6], [1, 11], [2, 17], [2, 12], [3, 24], [4, 28], [1,5], [2,6], [3,10], [4,15], [5,20], [4,27]])
# Y is the values: 1 high, 2 low
Y = np.array([1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1])

print(X)
print(Y)

plotx = X[:,0]
ploty = X[:,1]

plt.scatter(plotx, ploty)
plt.ylabel('RPM')
plt.show()

clf = GaussianNB()
clf.fit(X, Y)
pred = clf.predict(X)

print(clf.predict([[3, 25]]))

print(pred) 
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, Y))


Drop NaN values: file = file.dropna()


# REGRESSIONS
import sklearn
import quandl, math
import pandas as pd 
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# /Users/bddupont/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: 
# DeprecationWarning: This module was deprecated in version 0.18 in favor of 
# the model_selection module into which all the refactored classes and functions 
# are moved. Also note that the interface of the new CV iterators are different 
# from that of this module. This module will be removed in 0.20




df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low','Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['HL_PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'HL_PCT_change', 'Adj. Volume']]

# put Adj close prices in forecast col for now
forecast_col = 'Adj. Close'
#In machine learning you can't work with NaN data: instead of cutting it out, make it an outlier
df.fillna(value = -999999, inplace = True)

# forecast 10% out of the dataframe. math.ceil rounds up numbers
forecast_out = int(math.ceil(0.01*len(df)))

# shifts all rows: each row will be the adj close 10 days into the future
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace = True)

# FEATURES AND LABELS: X = features, Y = labels
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# X = X[:-forecast_out]
# X_lately = X[-forecast_out:]

# not mandatory
X = preprocessing.scale(X)


# Test dATA FOR 20% of data: 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#define classifider as linear regression
# can change algorithms to svm.SVR and leave everyhting the same
# allows you to test different algorithms
clf = LinearRegression()

#fit data (train)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

## REgression with Plot:
import quandl, math, datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

# Train and test on part of the dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
# use part of the data to train the classifier
clf.fit(X_train, y_train)
# use part of the data to test the classifier
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

print(confidence)

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

import quandl, math, datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

quandl.ApiConfig.api_key = "U5zALDafBdzSVuVmipmJ"

style.use('ggplot')

df = quandl.get("WIKI/AMZN")

df = df.tail(500)

# print(df.shape)

df = df.reset_index()
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.dropna()
# print(df.head(50))

#populate new fields with past values
for i in range(500,550):
    df.loc[i] = df.loc[i-100]

df['Labels'] = df.Close

# print(df.tail(60))

clf = LinearRegression()

y = np.array(df.Labels)
df = df[['Open', 'High', 'Low', 'Volume']]

X = np.array(df)

# print(X)
# print(y)

X_train = X[0:50]
X_test = X[51:101]
y_train = y[0:50]
y_test = y[51:101]

clf.fit(X_train,y_train)

#Pickling to save the trained classifier
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf,f)
pickle_in = open('linearregression.pickle', 'rb')
#clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

y[500:-1] = clf.predict(X[500:-1])

print(accuracy)

dates = np.array(df.index)

plt.plot(dates,y)
plt.show()



# COpy files:

import shutil
from shutil import copyfile
# with TEst 1 not present: the script will create Test1
source = "/Users/bddupont/Desktop/Test"
destination = "/Users/bddupont/Desktop/Test1"
shutil.copytree(source, destination)




