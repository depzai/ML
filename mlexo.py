# PANDAS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import svm


def importfile(source):
#    source = input('What is the path to the file?')
    print('path to the file:')
    print(source)
    global file
    file = pd.read_csv(source)
    print('Here are the first 10 lines:')
    print('\n')
    print(file.head(10))
    
#importfile()    

#/Users/bddupont/Desktop/BI/Python/Housing.csv

def explaindata():
    global dims
    dims = file.shape
    print('\n')
    print('the dimensions are (rows, columns)')
    print(dims)
    global columns
    columns = list(file)
    print('Here are the column headers:')
    print(columns)
    column_name = input('Which column do you want unique values for? ')
    print('\n')
    print(file[column_name].unique())

#explaindata()

#/Users/bddupont/Desktop/BI/Python/Housing.csv

#explaindata(file)

def pivotdata(cats, val, cat, cat2, meth):
#    cats = input('How many categories in the pivot table: 1 or 2? ')
    if cats == '1':
#        cat = input('Which category?')
#        val = input('Which values?')
#        meth = input('sum or count')
        if meth == 'sum':
            piv = pd.pivot_table(file,index=[cat],values=[val],aggfunc=np.sum)
        elif meth == 'count':
            piv = pd.pivot_table(file,index=[cat],values=[val],aggfunc=[len])
        else: print('only sum or count')
    elif cats == '2':
#        cat = input('Which category first?')
#        cat2 = input('Which other category?')
#        val = input('Which values?')
#        meth = input('sum or count')
        if meth == 'sum':
            piv = pd.pivot_table(file,index=[cat, cat2],values=[val],aggfunc=np.sum)
        elif meth == 'count':
            piv = pd.pivot_table(file,index=[cat, cat2],values=[val],aggfunc=np.count)
        else: print('only sum or count')
    else: print('only 1 or 2 categories')
    print('Pivot Table')
    print('\n')
    print(piv)

#pivotdata()
def info():
    print('Import function: importdata("path to file")')
    print('\n')
    print('Explain Funtion: explaindata()')
    print('\n')
    print('Pivot Table function: pivotdata(#categories, values column, category 1, category 2, sum or count)')

#info()
importfile('/Users/bddupont/Desktop/BI/Python/Downloaded/Canceldates.csv')
#explaindata()
file.date_cancel = file.date_cancel.fillna('long term')
file = file.sample(frac = 0.2)

file = file.drop(file[file.date_cancel == 'Jan-2016'].index)
file = file.drop(file[file.date_cancel == 'Feb-2016'].index)
file = file.drop(file[file.date_cancel == 'Mar-2016'].index)

# print(file.head(10))
# print(file.date_cancel.unique())

file['type'] = ('')

# print(file.head(10))

# print(file.shape)

file.loc[file.convert_al > 0, 'type'] = 'convert'
file.loc[file.direct_al > 0, 'type'] = 'direct'

# disctionary of label values
val = ['Apr-2016', 'May-2016', 'Jun-2016','Jul-2016', 'Aug-2016',
'Sep-2016', 'Oct-2016', 'Nov-2016', 'Dec-2016', 'Jan-2017',
'Feb-2017', 'Mar-2017', 'Apr-2017', 'May-2017', 'long term']
newval = {'Apr-2016': 1, 'May-2016': 2, 'Jun-2016': 3,'Jul-2016': 4, 'Aug-2016': 5,
'Sep-2016': 6, 'Oct-2016': 7, 'Nov-2016': 8, 'Dec-2016': 9, 'Jan-2017': 10,
'Feb-2017': 11, 'Mar-2017': 12, 'Apr-2017': 13, 'May-2017': 14, 'long term': 100 }

for i in range(0, len(val)):
    file.loc[file.date_cancel == val[i], 'date_cancel'] = newval[val[i]]

file = file.drop(file[file.date_cancel == 100].index)
file = file.drop(file[file.revenue == 0].index)

print(newval)

file.loc[file.type == 'convert', 'type'] = 1
file.loc[file.type == 'direct', 'type'] = 2

file.date_cancel = file.date_cancel.fillna(0)

print('Convert value is 1, direct value is 2')

labels = np.array(file.date_cancel)

# piv = pivotdata('1', 'convert_al', 'date_cancel', 'direct_al', 'sum')

# print(279854/file['convert_al'].sum())

dataset = file.drop('convert_al',1)
dataset = dataset.drop('direct_al',1)
dataset = dataset.drop('date_cancel',1)
dataset = dataset.drop('date_start',1)
dataset = dataset.drop('customer_key',1)

dataset.revenue = dataset.revenue.fillna(0)
dataset.type = dataset.type.fillna(0)

print(dataset.head(20))

dataarray = np.array(dataset)
# print(dataarray)

rev = np.array(dataset['revenue'])

print(rev)
print(labels)

dataplot = pd.DataFrame(rev,labels)
dataplot = dataplot.reset_index()
dataplot.columns = ['retention', 'revenue']


print(dataplot.head(20))
plt.scatter(dataplot.retention, dataplot.revenue)
plt.show()

#ML Starts
clf = svm.SVC(gamma = 0.01, C = 1)

x,y = dataarray[:-10], labels[:-10]

clf.fit(x,y)

pred = clf.predict(x)


print(clf.predict(x[-2]))
print('Real Value: ')
print(labels[-2])

from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y))



