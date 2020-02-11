

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
    plt.title(r'Data $\sigma_2^2 = $' + str(variance))
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

# Now plot the costs
fig= plt.figure(dpi= DPI, figsize= (8, 8))
plt.plot(variances, costs, lw= lw, c= 'b')
plt.title('Cost of clustering with different ' + r'$\sigma_2^2$')
plt.xlabel(r'$\sigma_2^2$')
plt.ylabel('Cost')
plt.xlim((0, num_iter))
plt.grid(True)
savefig(plt, "output/q2_cost.png")
plt.close()
