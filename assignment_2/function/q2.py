

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from library import *
from util import *
import params

num_dimensions = 2
k              = 2
#===============================================================================
# Execution starts here
#===============================================================================
data, labels, cluster_names = get_iris()

# Euclidean Distance
print("Doing MDS with euclidean...")
data_trans = get_MDS(data, n_components= num_dimensions, metric= "euclidean")
print("Running k-means with k= {}...".format(k))
cluster_centers_km, cluster_labels_km, _ = kmeans (data_trans , num_clusters= k)

fig= plt.figure(dpi= params.DPI, figsize= (24, 16))
plt.subplot(2,3,1)
plot_scatter(data_trans  , labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('MDS Euclidean (Color by species)')
plt.subplot(2,3,4)
plot_scatter(data_trans  , cluster_labels_km, loc= 'upper center', show_centers= True, cluster_centers= cluster_centers_km)
plt.title('k-Means (Color by cluster)')

# Manhattan/Cityblock Distance
print("Doing MDS with manhattan...")
data_trans = get_MDS(data, n_components= num_dimensions, metric= "cityblock")
print("Running k-means with k= {}...".format(k))
cluster_centers_km, cluster_labels_km, _ = kmeans (data_trans , num_clusters= k)

plt.subplot(2,3,2)
plot_scatter(data_trans  , labels           , loc= 'upper right', cluster_names_list= cluster_names)
plt.title('MDS Manhattan (Color by species)')

plt.subplot(2,3,5)
plot_scatter(data_trans  , cluster_labels_km, loc= 'upper right', show_centers= True, cluster_centers= cluster_centers_km)
plt.title('k-Means (Color by cluster)')

# Cosine distance
print("Doing MDS with cosine...")
data_trans = get_MDS(data, n_components= num_dimensions, metric= "cosine")
print("Running k-means with k= {}...".format(k))
cluster_centers_km, cluster_labels_km, _ = kmeans (data_trans , num_clusters= k)

plt.subplot(2,3,3)
plot_scatter(data_trans  , labels           , loc= 'upper right', cluster_names_list= cluster_names)
plt.title('MDS Cosine (Color by species)')

plt.subplot(2,3,6)
plot_scatter(data_trans  , cluster_labels_km, loc= 'upper right', show_centers= True, cluster_centers= cluster_centers_km)
plt.title('k-Means (Color by cluster)')

savefig(plt, "output/q2_mds.png", newline= True)
plt.close()
