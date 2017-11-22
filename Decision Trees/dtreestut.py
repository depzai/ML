import numpy as np 
import pandas as pd 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import sys

print(sys.version)
# import pydotplus
# import io

# from scipy import misc
# import seaborn as sns

# %matplotlib inline

# import
data = pd.read_csv('data.csv')

# print(data.describe())

# print(data.info())

# 'target' is whether user liked song or not (1 or 0)

train, test = train_test_split(data, test_size = 0.15)

print(train.shape)
# EDA to visualize data
print(data.head())

print("train size is {} and test size in {}".format(len(train), len(test)))

# Histogram of tempo the users likes (pos) and dislikes (neg)
pos_tempo = data[data['target'] == 1]['tempo']
neg_tempo = data[data['target'] == 0]['tempo']
pos_dance = data[data['target'] == 1]['danceability']
neg_dance = data[data['target'] == 0]['danceability']
pos_live = data[data['target'] == 1]['liveness']
neg_live = data[data['target'] == 0]['liveness']
pos_loud = data[data['target'] == 1]['loudness']
neg_loud = data[data['target'] == 0]['loudness']

# fig is the size of the chart
# fig = plt.figure(figsize=(12,8))
# plt.title('Song tempo like / dislike distribution')

# pos_tempo.hist(alpha = 0.7, bins = 30, label = 'positive', color = 'k')
# neg_tempo.hist(alpha = 0.7, bins = 30, label = 'negative', color = 'g')
# plt.legend(loc = 'upper right')
# plt.show()

# new figure that has subplots with new features
fig2 = plt.figure(figsize=(15,15))

#331 means 3 by 3 figure, position 1
ax3 = fig2.add_subplot(331)
ax3.set_xlabel('Danceability')
ax3.set_ylabel('Count')
ax3.set_title('Song Danceablity Like Distrib')
pos_dance.hist(alpha=0.5, bins  =30)
ax4 = fig2.add_subplot(331)
neg_dance.hist(alpha = 0.5, bins = 30)

ax5 = fig2.add_subplot(332)
ax5.set_xlabel('Tempo')
ax5.set_ylabel('Count')
ax5.set_title('Song tempo Like Distrib')
pos_tempo.hist(alpha=0.5, bins  =30)
ax6 = fig2.add_subplot(332)
neg_tempo.hist(alpha = 0.5, bins = 30)

ax7 = fig2.add_subplot(333)
ax7.set_xlabel('Liveness')
ax7.set_ylabel('Count')
ax7.set_title('Song liveness Like Distrib')
pos_live.hist(alpha=0.5, bins  =30)
ax8 = fig2.add_subplot(333)
neg_live.hist(alpha = 0.5, bins = 30)

ax9 = fig2.add_subplot(334)
ax9.set_xlabel('Liveness')
ax9.set_ylabel('Count')
ax9.set_title('Song loudness Like Distrib')
pos_loud.hist(alpha=0.5, bins  =30)
ax10 = fig2.add_subplot(334)
neg_loud.hist(alpha = 0.5, bins = 30)

# plt.show()

# train classifier
# min simple split: 2 is very specific to the training data
# so risk of overfit. 100 is more moderately specific (fewer leaves)
# means split after 100 observations, if 2, it create a lot of cases

clf = DecisionTreeClassifier(min_samples_split = 50)

features = ['danceability', 'loudness', 'valence', 'energy', 'instrumentalness', 'acousticness', 'speechiness', 'key', 'liveness']

X_train = train[features]
y_train = train['target']

X_test = test[features]
y_test = test['target']

dt = clf.fit(X_train, y_train)

pred = dt.predict(X_test)

target = np.array(y_test)

results = pd.DataFrame(pred, target)
results.reset_index(inplace = True)
results.columns = ['pred', 'target']

results['score'] = ''

for i in range(len(results)):
	if results['pred'][i] == results['target'][i]:
		results['score'][i] = 1
	else:
		results['score'][i] = 0

print(results.head(100))

score = results['score'].sum() / len(results['score'])

print("the accuracy score is {}".format("{:.2%}".format(score)))



dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features) 
graph = graphviz.Source(dot_data) 
graph.render("dectree")

# dot_data = tree.export_graphviz(clf, out_file=None,  feature_names=features,  
#     class_names=['like', 'dislike'],  filled=True, rounded=True, special_characters=True) 
# graph = graphviz.Source(dot_data)  
# graph.render('dtree_render',view=True)

# def show_tree(tree, features, path):
# 	f = io.StringIO()
# 	export_graphviz(tree, out_file=f, feature_names = features)
# 	pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
# 	img = miscimread(path)
# 	plt.rcParams['figure.figsize'] = (20,20)
# 	plt.imshow(img)

# show_tree(dt, features, 'dtree.png')

# predict target



