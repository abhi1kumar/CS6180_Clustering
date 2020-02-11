

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from util import *

DPI   = 200
msize = 100
fs    = 15
lw    = 2
xmin  = -5
xmax  = 6
matplotlib.rcParams.update({'font.size': fs})

data = readcsv_to_numpy("input/data2.csv")

num_clusters = 6
clusters     = np.arange(num_clusters)+1
cost_km = np.zeros((num_clusters,))

for i in range(num_clusters):
    cluster_centers_km, cluster_labels_km, cost_km[i] = kmeans (data, i+1)
    # Visualize and output the results
    fig= plt.figure(dpi= DPI, figsize= (8, 8))
    plot(data, cluster_centers_km, cluster_labels_km, 'upper right')
    plt.title('k-Means')
    plt.xlim([xmin, xmax])
    plt.ylim([-2  , xmax])

    savefig(plt, "output/q3_k_" + str(i+1) + ".png")
    plt.close()

fig= plt.figure(dpi= DPI, figsize= (8, 8))
plt.plot(clusters, cost_km, lw= lw, marker='s', c= 'b', label= 'k-Means Cost' )
plt.title(r'Cost of clustering versus $k$')
plt.xlabel(r'#Clusters $(k)$')
plt.ylabel('Cost of Clustering')
plt.xlim((0, num_clusters+1))
plt.grid(True)
plt.legend(loc= 'upper right')
savefig(plt, "output/q3_cost_with_k.png", newline= True)
plt.close()
