

import os, sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial.distance import cdist as dist
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from util import *

DPI   = 200
msize = 100
fs    = 15
lw    = 2
matplotlib.rcParams.update({'font.size': fs})

def kcenter(data, num_clusters= 2, metric= "euclidean"):
    """
        Implements a k-center algorithm
    """
    N = data.shape[0]

    # Initialize cluster centers to -1
    center_indices = -np.ones((num_clusters,), dtype=np.int32)   

    # Compute the distance between every pair of points    
    dist_mat = dist(data, data, metric= metric) # N x N

    # Compute centers
    for i in range(num_clusters):
        if i == 0:
            # Choose an id arbitrarily from N
            new_center_id = np.random.randint(N)
        else:
            # Choose maximum of the minimum distance of the previous center indices
            new_center_id = np.argmax(np.min(dist_mat[center_indices[0:i]], axis= 0))

        center_indices[i] = new_center_id

    # Get the labels of the points
    cluster_labels = np.argmin(dist_mat[center_indices], axis= 0)   

    # Get the cost of clustering
    cost = np.max(np.min(dist_mat[center_indices], axis= 0))

    return data[center_indices], cluster_labels, cost

def kmeans(data, num_clusters):
    """
        Implements k-means algorithm
        Reference https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
    """
    km = KMeans(n_clusters= num_clusters, init='random',  n_init= 1, max_iter= 300, 
        tol=1e-04, random_state=0)
    cluster_labels_km = km.fit_predict(data)

    cost = km.inertia_/data.shape[0]

    return km.cluster_centers_, cluster_labels_km, cost

def run_k_center_k_means(data, num_clusters= 3):
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
# Main function
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
