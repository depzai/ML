#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination

DETERMINE WHO WOULD SURVIVE BASED ON THE DATA

'''
df = pd.read_excel('titanic.xls')

df.drop(['name','body'],1, inplace = True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace = True)

def handle_non_numerical_data(column):
	unique_values = df[column].unique()
	dico = {}
	for i in range(len(unique_values)):
		dico[unique_values[i]] = i
	templist = list(df[column])
	newlist = []
	for j in range(len(templist)):
		newlist.append(dico[templist[j]])
	df[column] = newlist

handle_non_numerical_data('sex')
handle_non_numerical_data('home.dest')
handle_non_numerical_data('cabin')
handle_non_numerical_data('embarked')

print(df.head())














