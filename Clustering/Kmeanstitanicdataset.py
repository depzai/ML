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

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

print(df.head())

clf = KMeans(n_clusters = 2)
df1 = df.drop(['survived'], 1)
X = np.array(df1)
y = np.array(df['survived'])

print(X)

clf.fit(X)



#Accuracy:
predictions = []
for i in range(len(X)):
	pred = np.array(X[i])
	pred = pred.reshape(-1,len(X[i]))
	predix = clf.predict(pred)
	predictions.append(predix)


print(predictions)














