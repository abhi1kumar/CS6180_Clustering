

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle

import params

def plot_scatter(data, cluster_labels, loc= 'center right', show_centers= False,  cluster_centers= None, cluster_names_list= None):
    if show_centers:
        num_clusters = cluster_centers.shape[0]
    else:
        num_clusters = np.unique(cluster_labels).shape[0]

    rng = np.random.RandomState(0)
    cycol = cycle('rgbcmy')

    if cluster_names_list is not None:
        assert len(cluster_names_list) == num_clusters

    # Plot all the points with a different color
    for i in range(num_clusters):
        pts_index = cluster_labels == i

        if cluster_names_list is not None:
            label_text = cluster_names_list[i]
        else:
            label_text = "Cluster " + str(i)

        plt.scatter(data[pts_index, 0], data[pts_index, 1], c=next(cycol), s= params.msize//5, label= label_text)

    # Finally plot all the cluster centers with black
    if show_centers:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', c='k', s= params.msize)

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
