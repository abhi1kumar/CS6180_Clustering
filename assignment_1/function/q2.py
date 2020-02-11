

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import *

def getCluster(n, muX, varX, muY, varY):
    X = np.random.normal(muX, varX, n)
    Y = np.random.normal(muY, varY, n)

    return np.column_stack((X, Y))

DPI   = 200
msize = 100
n     = 500
lw    = 2
xmin  = -4
xmax  = 10

num_iter  = 3
costs     = np.zeros((num_iter, ))
variances = np.arange(num_iter) + 1

#===============================================================================
# (A) Breaking k-means
#===============================================================================
num_clusters = 2
C1 = getCluster(n, 0, 1, 0, 1)

for i in range(variances.shape[0]):
    variance = variances[i]
    C2 = getCluster(n, 3, variance, 3, 1)
    data = np.vstack((C1, C2))
    cluster_centers_km, cluster_labels_km, cost_km = kmeans (data, num_clusters)  
    costs[i] = cost_km

    fig= plt.figure(dpi= DPI, figsize= (16, 8))
 
    plt.subplot(1,2,1)
    plt.scatter(C1[:,0], C1[:,1], color= 'blue' , s= msize//6)
    plt.scatter(C2[:,0], C2[:,1], color= 'green', s= msize//6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Data $\sigma_{2x}^2 = $' + str(variance))
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plot(data, cluster_centers_km, cluster_labels_km, 'upper right')
    plt.title('k-Means')
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])

    savefig(plt, "output/q2_variance_" + str(variance) + ".png")
    plt.close()

# Now plot the costs with different variance
fig= plt.figure(dpi= DPI, figsize= (8, 8))
plt.plot(variances, costs, lw= lw, marker='s', c= 'b')
plt.title('Cost of clustering versus different ' + r'$\sigma_{2x}^2$')
plt.xlabel(r'$\sigma_{2x}^2$')
plt.ylabel('Cost of clustering')
plt.xlim((0, num_iter+0.2))
plt.grid(True)
savefig(plt, "output/q2_cost_with_variance.png", newline= True)
plt.close()

#===============================================================================
# (B) Get cost plots with different k
#===============================================================================
num_clusters = 6
clusters     = np.arange(num_clusters)+1
cost_km = np.zeros((num_clusters,))

for i in range(num_clusters):
    cluster_centers_km, cluster_labels_km, cost_km[i] = kmeans (data, i+1)
    # Plot
    fig= plt.figure(dpi= DPI, figsize= (8, 8))
    plot(data, cluster_centers_km, cluster_labels_km, 'upper right')
    plt.title('k-Means')
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])

    savefig(plt, "output/q2_variance_" + str(variance) + "_k_" + str(i+1) + ".png")
    plt.close()

fig= plt.figure(dpi= DPI, figsize= (8, 8))
plt.plot(clusters, cost_km, lw= lw, marker='s', c= 'b', label= 'k-Means Cost' )
plt.title(r'Cost of clustering versus $k$')
plt.xlabel(r'#Clusters $(k)$')
plt.ylabel('Cost of Clustering')
plt.xlim((0, num_clusters+1))
plt.grid(True)
plt.legend(loc= 'upper right')
savefig(plt, "output/q2_cost_with_k.png", newline= True)
plt.close()
