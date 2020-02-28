

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans

def get_pca(data, n_components= 2, center_data= True):
    """
        Computes the PCA of the data
        data = numpy array of shape N x D
    """
    N = data.shape[0]

    if center_data:
        # Center the data
        print("Centering the input data...")
        mean = np.mean(data, axis=0)[np.newaxis, :]
        data = data - mean

    # Compute the covariance matrix
    covariance = np.dot(data.transpose(), data)/ N # D x D

    # Compute the eigen values and eigen vectors
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    # The eigenvalues are not necessarily ordered.
    # column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    w, v = LA.eig(covariance)

    # Order the eigenvalues and corresponding eigenvectors in descending order
    order = np.argsort(-w)
    v = v[:, order]

    # THIS is sometimes needed since we need to match the function exactly.
    # Remember that if v is an eigen value, -v is also an eigen value.    
    v[:, 1] = -v[:, 1]

    # Transform data
    data_transformed = np.dot(data, v)

    return data_transformed[:, 0:n_components]

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
