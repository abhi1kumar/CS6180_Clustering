

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle

msize = 100
fs    = 15
matplotlib.rcParams.update({'font.size': fs})

def plot(data, cluster_centers, cluster_labels):
    num_clusters = cluster_centers.shape[0]
    rng = np.random.RandomState(0)
    cycol = cycle('bgrcm')
    
    # Plot all the points with a different color
    for i in range(num_clusters):
        pts_index = cluster_labels == i
        plt.scatter(data[pts_index, 0], data[pts_index, 1], c=next(cycol), s= msize//6, label=  "Cluster " + str(i))

    # Finally plot all the cluster centers with black
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', c='k', s= msize)
    plt.rc('axes', axisbelow=True)
    plt.grid(True)
    plt.legend(loc= 'center right')
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
