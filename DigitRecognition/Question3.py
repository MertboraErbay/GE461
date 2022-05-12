from doctest import master
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat4py
from sklearn.manifold import TSNE
import sammon as smn
import seaborn as sns

#data['labels'][i][0] corresponds to data['digits'][i]
directory = os.path.dirname(__file__)

data = mat4py.loadmat(directory + '/digits.mat')

labels = []
for item in data['labels']:
    labels.append(item[0])

labelset = pd.DataFrame(labels)
labelset.columns = ['labels']

keys = []
for i in range(0,400):
    keys.append('p' + str(i))

featureset = pd.DataFrame(data['digits'])
featureset.columns = keys


#Question 3:
#Sammon

[y, E] = smn.sammon(featureset,2,maxiter=110)

sdf = pd.DataFrame(y,columns=['dim1','dim2'])

sdf['labels'] = labelset

plt.figure(figsize=(16,10))
sns.scatterplot(x='dim1',y='dim2',hue="labels",palette=sns.color_palette("hls", 10),data=sdf,legend="full",alpha=0.3)
plt.show()


#t-SNE
tsne = TSNE(n_components=2, perplexity=40, n_iter=500)
results = tsne.fit_transform(featureset)

tsnedf = pd.DataFrame()

tsnedf['labels'] = labelset
tsnedf['dim1'] = results[:,0]
tsnedf['dim2'] = results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(x = 'dim1', y = 'dim2',hue="labels",palette=sns.color_palette("hls", 10),data=tsnedf,legend="full",alpha=0.3)
plt.show()

    