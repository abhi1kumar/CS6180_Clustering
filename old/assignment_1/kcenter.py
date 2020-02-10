import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import astype
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import as_float_array
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.random import choice
from sklearn.externals.six import string_types

def k_centers(X, n_clusters, seed=None, metric='euclidean', metric_kw=None,
              verbose=False, random_state=None, copy_x=True):
    """K-centers clustering algorithm.
    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form as well as the number of
        centers to generate.
    seed : int
        The index of the initial point to choose as a cluster center. If None,
        defaults to a randomly-selected point.
    metric : string or callable
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:
        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    metric_kw : keyword arguments for the metric
    verbose : boolean, optional
        Verbosity mode.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the first center. Ignored unless
        seed=None. If an integer is given, it fixes the seed. Defaults to the
        global numpy random number generator.
    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    Returns
    -------
    center_indices : integer ndarray with shape (k,)
        Indices of centers found by k-centers.
    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the center the
        i'th observation is closest to.
    clust_size : float
        The final value of the size criterion (max distance to the closest
        center over all observations in the training set).
    """
    random_state = check_random_state(random_state)

    X = as_float_array(X, copy=copy_x)

    # # subtract mean of x for more accurate distance computations
    # if not sp.issparse(X) or hasattr(init, '__array__'):
    #     X_mean = X.mean(axis=0)
    # if not sp.issparse(X):
    #     # The copy was already done above
    #     X -= X_mean

    n_samples, n_features = X.shape

    if seed is None:
        cur_id = random_state.randint(n_samples)
    else:
        cur_id = seed

    if metric_kw is None:
        metric_kw = dict()

    # init
    if verbose:
        print("Initialization complete")

    center_indices = np.empty((n_clusters,), dtype=np.int32)
    center_indices.fill(-1)
    mindist = np.empty(n_samples, np.float64)
    mindist.fill(np.infty)
    labels = np.empty(n_samples, dtype=np.int32)
    labels.fill(-1)
    for center_id in range(n_clusters):
        center_indices[center_id] = cur_id
        dist = \
            pairwise_distances(
                X[cur_id].reshape(1,-1), X,
                metric=metric, **metric_kw).squeeze()
        labels[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)
        cur_id = mindist.argmax()

    clust_size = mindist.max()

    if verbose:
        print("Cluster Size %.3f" % clust_size)

    # if not sp.issparse(X):
    #     if not copy_x:
    #         X += X_mean

    return center_indices, labels, clust_size



class KCenters(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Centers clustering
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centers to generate.
    seed : int
        The index of the initial point to choose as a cluster center. If None,
        defaults to a randomly-selected point.
    metric : string or callable
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:
        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    metric_kw : keyword arguments for the metric
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    verbose : int, default 0
        Verbosity mode.
    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    Attributes
    ----------
    cluster_centers_ : array, (n_clusters, n_features)
        Coordinates of cluster centers
    cluster_center_indices_ : array, (n_clusters,)
        Indices of cluster centers in X
    labels_ :
        Labels of each point
    cluster_size_ : float
        Max distance of samples to their closest cluster center.
    Notes
    ------
    The k-centers problem is solved using Gonzalez's algorithm.
    The average complexity is given by O(k n), were n is the number of
    samples.
    """

    def __init__(self, n_clusters=8, seed=None, metric='euclidean',
                 metric_kw=None, verbose=0, random_state=None, copy_x=True):

        self.n_clusters = n_clusters
        self.seed = seed
        self.metric = metric
        self.metric_kw = metric_kw
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                        warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, y=None):
        """Compute k-centers clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_center_indices_, self.labels_, self.cluster_size_ = \
            k_centers(
                X, n_clusters=self.n_clusters, seed=self.seed,
                metric=self.metric, metric_kw=self.metric_kw,
                verbose=self.verbose, random_state=random_state,
                copy_x=self.copy_x)
        self.cluster_centers_ = X[self.cluster_center_indices_,:]
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        X = self._check_fit_data(X)
        return self.fit(X)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return \
            pairwise_distances(
                X, self.cluster_centers_, metric=self.metric,
                metric_kwargs=self.metric_kw)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return \
            pairwise_distances_argmin_min(
                self.cluster_centers_, X, metric=self.metric,
                metric_kwargs=self.metric_kw)[0]

    def score(self, X, y=None):
        """Opposite of the value of X on the K-centers objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        Returns
        -------
        score : float
            Opposite of the value of X on the K-centers objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return \
            -pairwise_distances_argmin_min(
                self.cluster_centers_, X, metric=self.metric,
                metric_kwargs=self.metric_kw)[0]
