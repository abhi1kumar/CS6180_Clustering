

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

print("Doing TSNE...")
data_trans = get_TSNE(data, n_components= num_dimensions)
print("Running k-means with k= {}...".format(k))
cluster_centers_km, cluster_labels_km, _ = kmeans (data_trans , num_clusters= k)

fig= plt.figure(dpi= params.DPI, figsize= params.size)
plt.subplot(1,2,1)
plot_scatter(data_trans  , labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('TSNE (Color by species)')

plt.subplot(1,2,2)
plot_scatter(data_trans  , cluster_labels_km, loc= 'upper center', show_centers= True, cluster_centers= cluster_centers_km)
plt.title('k-Means (Color by cluster)')

savefig(plt, "output/q3_tsne.png", newline= True)
plt.close()
