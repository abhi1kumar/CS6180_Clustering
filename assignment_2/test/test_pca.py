import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from function.library import *
from function.util import *
import function.params as params

num_dimensions = 2
k = 2
#===============================================================================
# Execution starts here
#===============================================================================
data, labels, cluster_names = get_iris()

data_trans = get_pca(data, n_components= num_dimensions)
data_trans_2 = get_pca(data, n_components= num_dimensions, scaling= False)

# Inbuilt function
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
pca_obj = PCA(n_components= num_dimensions,  svd_solver= 'full')
data_trans_3 = pca_obj.fit_transform(data)

fig= plt.figure(dpi= params.DPI, figsize= (18,6))
plt.subplot(1,3,1)
plot_scatter(data_trans  , labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('PCA with Scaling')

plt.subplot(1,3,2)
plot_scatter(data_trans_2, labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('PCA w/o  Scaling')

plt.subplot(1,3,3)
plot_scatter(data_trans_3, labels           , loc= 'upper center', cluster_names_list= cluster_names)
plt.title('Scipy PCA')

savefig(plt, "output/test_pca.png", newline= True)
plt.close()
