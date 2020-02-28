

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from library import *
from util import *

import params

num_dimensions = 2
k = 2
#===============================================================================
# Execution starts here
#===============================================================================
data_full     = load_iris()
data          = data_full.data
labels        = data_full.target
cluster_names = list(data_full.target_names)
print(data.shape)

#pca_obj = PCA(n_components= num_dimensions)
#data_trans = pca_obj.fit_transform(data)

print("Doing PCA...")
data_trans = get_pca(data, n_components= num_dimensions)
print("Running k-means with k= {}...".format(k))
cluster_centers_km, cluster_labels_km, _ = kmeans (data_trans , num_clusters= k)

fig= plt.figure(dpi= params.DPI, figsize= params.size)
plt.subplot(1,2,1)
plot_scatter(data_trans  , labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('PCA with Centering (Color by species)')

plt.subplot(1,2,2)
plot_scatter(data_trans  , cluster_labels_km, loc= 'upper center', show_centers= True, cluster_centers= cluster_centers_km)
plt.title('k-Means (Color by cluster)')

savefig(plt, "output/q1_centering_k_means.png", newline= True)
plt.close()

print("Doing PCA without centering...")
data_trans_2 = get_pca(data, n_components= num_dimensions, center_data= False)
fig= plt.figure(dpi= params.DPI, figsize= params.size)
plt.subplot(1,2,1)
plot_scatter(data_trans  , labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('PCA with Centering (Color by species)')

plt.subplot(1,2,2)
plot_scatter(data_trans_2, labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('PCA w/o Centering (Color by species)')

savefig(plt, "output/q1_with_and_without_centering.png", newline= True)
plt.close()
