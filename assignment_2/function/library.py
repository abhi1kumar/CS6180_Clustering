

"""
    Libraries consisting of commonly used functions
"""
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist as dist

def get_pca(data, n_components= 2, center_data= True, scaling= False):
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
    # of absolute value of eigenvalues
    order = np.argsort(-np.abs(w))
    v = v[:, order]
    w = w[order]

    # THIS is sometimes needed since we need to match the function exactly.
    # Remember that if v is an eigen value, -v is also an eigen value.    
    v[:, 1] = -v[:, 1]

    # Transform data
    # w is a 1-D array. Make its copies first and then do elementwise multiplication
    # to get the new basis
    w_array = np.repeat(w[np.newaxis, :], v.shape[0], 0)
    if scaling:
        new_basis = np.multiply(np.sqrt(w_array), v)
    else:
        new_basis = v

    data_transformed = np.dot(data, new_basis)

    # Dim reduced data
    data_reduced = data_transformed[:, 0:n_components]

    # Return the real components. Data can become complex because of complex 
    # eigenvectors
    return np.real(data_reduced)

def get_inbuilt_pca(data, n_components= 2):
    """
        Inbuilt function
        Reference 
        https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    """
    pca_obj = PCA(n_components= n_components,  svd_solver= 'full')
    data_reduced = pca_obj.fit_transform(data)

    return data_reduced

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

def get_MDS(data, n_components= 2, metric= "euclidean"):
    """
        Calculates the Multi-Dimensional Scaling algorithm using different
        distances
    """
    # Compute the distance between every pair of points    
    dist_mat = dist(data, data, metric= metric) # N x N

    # Calculate the matrix H
    N = data.shape[0]
    H = np.eye(N) - np.ones((N, N))/N

    # Apply double centering
    S = -0.5 * np.dot(np.dot(H, np.power(dist_mat, 2)), H)

    # Get PCA
    return get_pca(S, n_components= n_components, center_data= False)

def get_TSNE(data, n_components= 2):
    tsne = TSNE(n_components= n_components, verbose=1, perplexity= 40, n_iter= 300)
    data_reduced = tsne.fit_transform(data)

    return data_reduced
