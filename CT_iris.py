# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:20:36 2020

@author: Adithya Madhavan
"""

#Clustering using Iris Dataset
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns

df = data('iris')
df.head()
df.Species.value_counts()

df1 =  df.select_dtypes(exclude=['object'])
df1.head()

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
df1_scaled = scalar.fit_transform(df1)
type(df1_scaled)
type(df1_scaled)
#data2_scaled.describe() #it converts to different format
#pd.DataFrame(data2_scaled).describe()
#pd.DataFrame(data2_scaled).head()


#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  #hyper parameters

kmeans.fit(df1_scaled)
kmeans.inertia_  #sum of sq distances of samples to their centeroid
kmeans.cluster_centers_
kmeans.labels_
df1.shape
kmeans.n_iter_  #iterations to stabilise the clusters
kmeans.predict(df1)

clusterNos = kmeans.labels_
clusterNos
type(clusterNos)

df1.groupby([clusterNos]).mean()
df.groupby(['Species']).mean()

#%%

#hierarchical clustering
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))

plt.figure(figsize = (10,7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))
plt.axhline(y=15, color='r', linestyle='--')
plt.show();

#another method for Hcluster from sklearn
from sklearn.cluster import AgglomerativeClustering
aggCluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
aggCluster.fit_predict(df1_scaled)
aggCluster
aggCluster.labels_

