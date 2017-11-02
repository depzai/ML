# PANDAS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np




#IMport locally from same folder
df = pd.read_csv('/Users/bddupont/Desktop/BI/Python/results.csv')
#print(df.head(10))

#List unique values in the df['name'] column
print(df.sale_type_name.unique())    

#drop rows with Exclude and ALC
df = df.drop(df[df.sale_type_name == 'EXCLUDE'].index)
df = df.drop(df[df.sale_type_name == 'ALC'].index)

#print(df.head(50))

print(df.sale_type_name.unique())  

#EXTRACT DISTINCT CUSTOMERS AND THEIR JUL AND AUG SPEND
#custlist = pd.pivot_table(df,index=["customer_key"], columns = ["yr_mth_name"],values=["sum"],aggfunc=np.sum, fill_value = 0)

jul = df.loc[df['yr_mth_name'] == 'Jul-2017']

aug = df.loc[df['yr_mth_name'] == 'Aug-2017']

print(jul.head(12))

jul = jul.groupby('customer_key')[['sum']].sum()
jul = jul.reset_index()
jul.columns = ['customers', 'jul']

aug = aug.groupby('customer_key')[['sum']].sum()
aug = aug.reset_index()
aug.columns = ['customers', 'aug']

print(jul.head())
print(aug.head())

#MERGE DATA IN ONE TABLE
jul_aug = pd.merge(jul, aug, how = 'outer', on = 'customers')
jul_aug = jul_aug.fillna(0)

print(jul_aug.head(10))

#HIGHEST SPENDERS IN JUL
jul_aug['%Jul'] = ""
jul_rev = jul_aug['jul'].sum()

jul_aug['%Jul'] = jul_aug['jul']/jul_rev

jul_aug = jul_aug.sort_values(by = 'jul', ascending = 0)
jul_aug = jul_aug.reset_index()
jul_aug = jul_aug.drop('index',1)

print(jul_aug.head(50))

# ADD Cumulative value
cumul = np.array(jul_aug['jul'])
cumul = np.cumsum(cumul)/jul_rev

jul_aug['cumul%'] = cumul

print(jul_aug.head(50))

# Select top 10% of spenders
top_10 = jul_aug.loc[jul_aug['cumul%'] <= 0.1]

print(top_10.head(50))

print(top_10.shape)

top_10.head(20).to_html('/Users/bddupont/Desktop/BI/Python/Examples/exercise.html')


#print(df.sale_type_name.unique())


#custlist = pd.DataFrame(list(df.customer_key.unique()))

#custlist.columns = ['Customers']
#custlist['Revenue'] = ''

#print(custlist.shape)


#i = 0 #iterator for loop

#for i in range(0,custlist.shape[0]):
#    custlist.loc[custlist.index[i], 'Revenue'] = df.loc[df['customer_key'] == custlist.loc[custlist.index[i], 'Customers'], 'sum'].sum()

#print(custlist.head())


#custlist.loc[custlist.index[0], 'Revenue'] = 98


#for row in range(df.shape[0]): # df is the DataFrame
#         for col in range(df.shape[1]):
#             if df.get_value(row,col) == 'security_id':
#                 print(row, col)
#                 break


#SUmif:
#df.loc[(df['cohort_date'] == "Mar-11") & (df['country'] == "US"), 'revenue'].sum()