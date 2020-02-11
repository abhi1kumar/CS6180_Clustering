

import numpy as np
from scipy.spatial.distance import cdist as dist
from sklearn.cluster import KMeans

import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle

msize = 100
fs    = 15
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
    km = KMeans(n_clusters= num_clusters,  n_init= 1, max_iter= 300, 
        tol=1e-04, random_state= 0)
    cluster_labels_km = km.fit_predict(data)

    cost = km.inertia_/data.shape[0]

    return km.cluster_centers_, cluster_labels_km, cost

def plot(data, cluster_centers, cluster_labels, loc= 'center right'):
    num_clusters = cluster_centers.shape[0]
    rng = np.random.RandomState(0)
    cycol = cycle('bgrcmy')
    
    # Plot all the points with a different color
    for i in range(num_clusters):
        pts_index = cluster_labels == i
        plt.scatter(data[pts_index, 0], data[pts_index, 1], c=next(cycol), s= msize//6, label=  "Cluster " + str(i))

    # Finally plot all the cluster centers with black
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', c='k', s= msize)
    plt.rc('axes', axisbelow=True)
    plt.grid(True)
    plt.legend(loc= loc)
    plt.xlabel('x')
    plt.ylabel('y')

def savefig(plt, path, show_message= True, tight_flag= True, newline= False):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(path)
    if newline:
        print("")
