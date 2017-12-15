# PANDAS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


def importfile():
    source = input('What is the path to the file?')
    print('path to the file:')
    print(source)
    global file
    file = pd.read_csv(source)
    print('Here are the first 10 lines:')
    print('\n')
    print(file.head(10))
    
#importfile()    

#/Users/bddupont/Desktop/BI/Python/housing.csv

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

info()
importfile()
explaindata()






