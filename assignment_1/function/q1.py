

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import *

DPI   = 200
msize = 100
fs    = 15
lw    = 2
matplotlib.rcParams.update({'font.size': fs})

def run_k_center_k_means(data, num_clusters= 3):
    """
        Runs k-center and k-means on the same data and plots the outputs side by
        side
    """
    cluster_centers_kc, cluster_labels_kc, _ = kcenter(data, num_clusters)
    cluster_centers_km, cluster_labels_km, _ = kmeans (data, num_clusters)

    fig= plt.figure(dpi= DPI, figsize= (16, 8))

    plt.subplot(1,2,1)
    plot(data, cluster_centers_kc, cluster_labels_kc)
    plt.title('k-Center')

    plt.subplot(1,2,2)
    plot(data, cluster_centers_km, cluster_labels_km)
    plt.title('k-Means')

    savefig(plt, "output/q1_num_clusters_" + str(num_clusters) + ".png")
    plt.close()

#===============================================================================
# Execution starts here
#===============================================================================
# Read the data from csv
# The zeroth column should be used as index
data = pd.read_csv("input/data1.csv", index_col= 0).to_numpy()

run_k_center_k_means(data, num_clusters= 3)
run_k_center_k_means(data, num_clusters= 4)

#Get cost plots
num_clusters = 50
clusters     = np.arange(num_clusters)+1
cost_kc = np.zeros((num_clusters,))
cost_km = np.zeros((num_clusters,))

for i in range(num_clusters):
    _, _ , cost_kc[i] = kcenter(data, i+1)
    _, _ , cost_km[i] = kmeans (data, i+1)

fig= plt.figure(dpi= DPI, figsize= (8, 8))
plt.plot(clusters, cost_kc, lw= lw, c= 'b', label= 'k-Center')
plt.plot(clusters, cost_km, lw= lw, c= 'g', label= 'k-Means' )
plt.title('Cost of clustering')
plt.xlabel('#Clusters')
plt.ylabel('Cost')
plt.xlim((0, num_clusters))
plt.grid(True)
plt.legend(loc= 'upper right')
savefig(plt, "output/q1_cost.png")
plt.close()
