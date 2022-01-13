# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:54:58 2021

@author: lucas
"""

import numpy as np
from tabulate import tabulate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

test_data = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStest.dt', delimiter =',')
train_data = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStrain.dt', delimiter =',')

x_test = test_data[:, 0:62]
y_test = test_data[:, -1]
x_train = train_data[:, 0:62]
y_train = train_data[:, -1]

un = np.unique(y_train, return_counts=True)
y_selected = un[0][un[1]>64]

x_test = test_data[np.in1d(y_test,y_selected).nonzero(), 0:62][0]
y_test = test_data[np.in1d(y_test,y_selected).nonzero(), -1][0]
x_train = train_data[np.in1d(y_train,y_selected).nonzero(), 0:62][0]
y_train = train_data[np.in1d(y_train,y_selected).nonzero(), -1][0]

print(tabulate(np.unique(y_test, return_counts=True), tablefmt="latex", floatfmt=".2f"))

print(tabulate(np.unique(y_train, return_counts=True), tablefmt="latex", floatfmt=".2f"))

def f_norm(x, mean_vec, variance_vec):
    x = (x-mean_vec) / variance_vec
    return x

x_var = np.std(x_train, axis = 0)
x_mean = np.mean(x_train, axis = 0)

x_train_trans = f_norm(x_train, x_mean, x_var)
x_test_trans = f_norm(x_test, x_mean, x_var)

# PCA
pca_trans = PCA()
pca_trans.fit(x_train_trans)

components_2 = pca_trans.components_[0:2, :]
pc1 = x_train_trans @ components_2[0, :]
pc2 = x_train_trans @ components_2[1, :]

to_plot = pd.DataFrame(np.array([pc1, pc2, y_train.astype(int)]).T,
                   columns=['pc1', 'pc2', 'target'])

sns_plot = sns.scatterplot(x=to_plot["pc1"], y=to_plot["pc2"], hue=to_plot["target"].astype(str), alpha = 0.8)

sns_plot.figure.savefig("C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/plots/pca_2components.png")

# 4 means and 4 means ++
kmeans = KMeans(n_clusters=4, init = 'random').fit(x_train_trans)
kmeanspp = KMeans(n_clusters=4, init = 'k-means++').fit(x_train_trans)
kmeans_centers = pd.DataFrame(np.array([kmeans.cluster_centers_ @ components_2[0, :],
                              kmeans.cluster_centers_ @ components_2[1, :], 
                              ['4-means','4-means','4-means','4-means']]).T,
                              columns=['pc1','pc2','k_means_type'])
kmeanspp_centers = pd.DataFrame(np.array([kmeanspp.cluster_centers_ @ components_2[0, :],
                              kmeanspp.cluster_centers_ @ components_2[1, :],
                              ['4-means++','4-means++','4-means++','4-means++']]).T,
                              columns=['pc1','pc2','k_means_type'])

fig, ax = plt.subplots(1, 1, figsize=(7,7))

sns.scatterplot(x=to_plot["pc1"], y=to_plot["pc2"], hue=to_plot["target"].astype(str), alpha = 0.6, ax=ax)
plt.scatter(x = kmeans_centers["pc1"].astype(float), y=kmeans_centers["pc2"].astype(float), 
            label = "4-means", color = 'm', marker = '^')
plt.scatter(x = kmeanspp_centers["pc1"].astype(float), y=kmeanspp_centers["pc2"].astype(float), 
            label = "4-means++", color = 'c', marker = '+')
plt.legend()
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")

plt.show()
fig.savefig("C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/plots/4means.png")





