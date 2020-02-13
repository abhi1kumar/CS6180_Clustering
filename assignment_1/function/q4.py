

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

from util import *
INF = 100
DPI = 200

def get_key(dic, n):
    """
        Returns keys of the dictionary after sorting
        Reference- https://stackoverflow.com/a/16977466
    """
    try:
       return [key for (key, value) in sorted(dic.items())][n] #list(dic)[n]
    except IndexError:
       print("not enough keys")

def get_value(dic, n):
    """
        Returns values of the dictionary after sorting
    """
    try:
        return [value for (key, value) in sorted(dic.items())][n]
    except IndexError:
        print("not enough keys")

def distance_bet_clusters(ind1, ind2, data, linkage= "single", metric= "euclidean"):  
    ind_all = np.concatenate([ind1, ind2])  
    dist_mat = dist(data[ind_all], data[ind_all], metric= metric)

    # Look only in the rows 0:ind2 and in the columns 0:ind1
    len1 = ind1.shape[0]
    len2 = ind2.shape[0]
    dist_useful = dist_mat[len1:, 0:len1]

    if linkage== "single":
        dist_cluster = np.min(dist_useful.flatten())
    else:
        raise NotImplementedError("To be implemented")

    return dist_cluster

def get_closest_nodes(nodes_dict, data):
    num_nodes = len(nodes_dict) 
    dist      = np.zeros((num_nodes, num_nodes))
    min_dist  = INF
    closest_nodes_index = np.zeros((2,), dtype= np.uint8)

    for i in range(num_nodes):
        for j in range(i+1):
            if i == j:
                # Assign distance between two clusters with same index as infinity
                # so that this trivial solution is avoided while calculating minimum
                dist[i,j] = INF
            else:
                dist[i,j] = distance_bet_clusters(get_value(nodes_dict, i), get_value(nodes_dict, j), data)
                if min_dist > dist[i, j]:
                    min_dist = dist[i, j]
                    closest_nodes_index[0] = i
                    closest_nodes_index[1] = j

    return min_dist, closest_nodes_index

#===============================================================================
# Execution starts here
#===============================================================================
data = readcsv_to_numpy("input/data2.csv")
# data = np.array([[1,2],[2,2],[3,4],[4,4],[5,6],[6,6],[7,8],[8,8],[9,10],[10,10]])

N = data.shape[0]

# Make a dictionary that keeps track of the merging and stuff
# Initially each element is a separate node
nodes_dict = {}
for i in range(N):
    nodes_dict[i] = np.array([i])

# This keeps track of the number of clusters.
cluster_cnt = N-1

# The scipy dendogram function takes the following format to display
# Each row of the linkage variable stores 
# Cluster_index_0 to be merged, Cluster_index_1 to be merged, distance, number_of_points_in_new_cluster
# We will store it in a list and later conver to numpy array
linkage = []

while len(nodes_dict) > 1:
    min_dist, closest_nodes_index = get_closest_nodes(nodes_dict, data)
    cluster_ind_0 = get_key(nodes_dict, closest_nodes_index[0])
    cluster_ind_1 = get_key(nodes_dict, closest_nodes_index[1])
    num_nodes_new_cluster = nodes_dict[cluster_ind_0].shape[0] + nodes_dict[cluster_ind_1].shape[0]

    # Add information of the new cluster
    linkage.append(np.array([cluster_ind_0, cluster_ind_1, min_dist, num_nodes_new_cluster]))

    # Increase the cluster counter
    cluster_cnt += 1
    
    # Concatenate indices into new array
    new_indices = np.concatenate([nodes_dict[cluster_ind_0], nodes_dict[cluster_ind_1]])
    
    # Delete the old nodes from nodes_dict
    del nodes_dict[cluster_ind_0]
    del nodes_dict[cluster_ind_1]

    # Add the new node to the nodes_dict
    nodes_dict[cluster_cnt] = new_indices

    curr_num_clusters = len(nodes_dict)

    if cluster_cnt%100 == 0 or curr_num_clusters == 1:
        print("{} clusters done...".format(cluster_cnt))

    # Plot and save scatter when curr_num_clusters < 5
    if curr_num_clusters < 5 and curr_num_clusters > 1:
        cluster_labels = np.zeros((N, ), dtype= np.uint8)
        for i in range(curr_num_clusters):
            indices = get_value(nodes_dict, i)
            cluster_labels[indices] = i

        fig= plt.figure(dpi= DPI, figsize= (8, 8))        
        plot(data, np.zeros((curr_num_clusters,)), cluster_labels, loc= "upper right", show_centers= False)
        plt.title('HAC k= ' + str(curr_num_clusters))
        savefig(plt, "output/q4_k_" + str(curr_num_clusters) + ".png")

# Convert linkage to a numpy array
linkage = np.array(linkage)

fig= plt.figure(dpi= DPI, figsize= (8, 8))
dendrogram(linkage, orientation= 'top', distance_sort= 'descending', show_leaf_counts= True)
plt.xlabel("Data point index")
plt.ylabel("Distance (unit)")
plt.title("Dendogram")
savefig(plt, "output/q4.png", newline= True)
plt.close()
