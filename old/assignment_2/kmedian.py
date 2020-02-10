import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES

def _validate_medoid_shape(X, k, medoids):
    """Check if medoids is compatible with X and k"""
    if len(medoids) != k:
        raise ValueError('The shape of the initial medoids (%s) '
                         'does not match the number of clusters %i'
                         % (medoids.shape, k))
    if medoids.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial medoids %s "
            "does not match the number of features of the data %s."
            % (medoids.shape[1], X.shape[1]))


def _init_medoids(X, k, init=None, random_state=None):
    """Compute the initial medoids
    Parameters
    ----------
    X: array, shape (n_samples, n_features)
    k: int
        number of medoids
    init: {None or ndarray or callable} optional
        Method for initialization
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the medoids. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    Returns
    -------
    medoid_indices: array, shape = (k,)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if init is None:
        medoid_indices = random_state.permutation(n_samples)[:k]
    elif hasattr(init, '__array__'):
        medoid_indices = init
    elif callable(init):
        medoid_indices = init(X, k, random_state=random_state)
    else:
        raise ValueError("the init parameter for k-medians should "
                         "be None or an ndarray; "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(medoid_indices):
        medoid_indices = medoid_indices.toarray()

    return medoid_indices



def local_search_k_medians(X, n_clusters, metric='euclidean',
                           metric_kwargs=None, init=None, max_iter=300,
                           verbose=False, tol=1e-2, random_state=None,
                           return_n_iter=False, precompute_distances='auto'):
    """K-medians clustering algorithm.
    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form as well as the number of
        medoids to generate.
    metric : string or callable, default 'euclidean'
        This gets passed to pairwise_distances and determines which metric is
        to be used for the algorithm.
    metric_kwargs : dict
        Keyword arguments for the metric.
    max_iter : int, optional, default 300
        Maximum number of iterations of the k-medians algorithm to run.
    init : {None, or ndarray, or a callable}, optional
        Method for initialization, default to 'k-medians++':
        None : generate k medoids sampled (without replacement) from the data.
        If an ndarray is passed, it should be of shape (n_clusters,)
        and gives the indices of the initial medoids.
        If a callable is passed, it should take arguments X, k and
        and a random state and return initial indices.
    tol : float, optional
        The relative reduction in the cost before declaring convergence.
    verbose : boolean, optional
        Verbosity mode.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the medoids. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    return_n_iter : bool, optional
        Whether or not to return the number of iterations.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    Returns
    -------
    medoid_indices : integer ndarray with shape (n_clusters,)
        Medoids found at the last iteration of k-medians same as input
        parameter unless parameter is None.
    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the medoid the
        i'th observation is closest to.
    cost : float
        The final value of the cost criterion (sum of squared distances to
        the closest medoid for all observations in the training set).
    n_iter: int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    cost_history: float ndarray with shape (n_iter+1,)
        Value of cost criterion after each iteration.
        Returned only if `return_n_iter` is set to True.
    """
    random_state = check_random_state(random_state)

    n_samples = X.shape[0]
    X = as_float_array(X)
    break_iter = 0

    if precompute_distances == 'auto':
        precompute_distances = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distances, bool):
        pass
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         precompute_distances)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    if metric_kwargs is None:
        metric_kwargs = dict()

    # Initialize medoids
    medoid_indices = _init_medoids(X, n_clusters, init=init,
                                   random_state=random_state)
    if verbose:
        print('Initial indices:')
        print(medoid_indices)
        print('Initial medoids:')
        print X[medoid_indices]

    # Initialize cost
    if precompute_distances:
        distances = \
            pairwise_distances(
                X, X[medoid_indices], metric=metric,
                **metric_kwargs)
        costs = distances.min(axis=1)
    else:
        costs = \
            pairwise_distances_argmin_min(
                X, X[medoid_indices], metric=metric, **metric_kwargs)[1]
    best_cost = costs.sum()

    # Get array of the other indices by subtracting the current medoid indices
    # from all integers in range
    other_indices = \
        np.setdiff1d(np.arange(n_samples), medoid_indices.copy(),
                     assume_unique=True)

    if verbose:
        print('Initial cost: {:.2g}'.format(best_cost))
        print("Initialization complete")

    clean_cluster = 0
    
    cost_history = [best_cost]

    for i in range(max_iter):
        for med in range(n_clusters):

            medoid_cost = np.infty

            # Try the other points as this medoid
            for oth in range(n_samples-n_clusters):
                oidx = other_indices[oth]
                # If we precomputed the distances, update just the one
                if precompute_distances:
                    distances[:,med] = \
                        pairwise_distances(
                            X, X[oidx][np.newaxis,:],
                            metric=metric, **metric_kwargs).squeeze()
                    distances.min(axis=1, out=costs)
                # Otherwise just get the costs efficiently
                else:
                    tmp_indices = medoid_indices.copy()
                    tmp_indices[med] = oidx
                    # Compute costs on the fly
                    costs = \
                        pairwise_distances_argmin_min(
                            X, X[tmp_indices], metric=metric,
                            **metric_kwargs)[1]
                cost = costs.sum()

                # Only bother updating if we improve the cost
                if cost < medoid_cost:
                    medoid_cost = cost
                    medoid_idx = oth

            # Use tolerance here to ensure polynomial runtime
            if medoid_cost < (1-tol)*best_cost:
                clean_cluster = 0
                # Swap indices
                other_indices[medoid_idx], medoid_indices[med] = \
                    medoid_indices[med], other_indices[medoid_idx]
                best_cost = medoid_cost
                if verbose:
                    print('New medoid {:d} with cost {:.2f}'\
                        .format(med, best_cost))
            else:
                clean_cluster += 1

            # Now that we know which one is the right medoid, update the
            # distances matrix (if precomputing)
            if precompute_distances:
                distances[:,med] = \
                    pairwise_distances(
                        X, X[medoid_indices[med]][np.newaxis,:], metric=metric,
                        **metric_kwargs).squeeze()

            # If we've had n_cluster clean clusters in a row, then we don't need
            # to continue; regardless of state of med
            if clean_cluster >= n_clusters:
                break

        if verbose:
            print("Iteration %3d, cost %.3f" % (i, best_cost))

        if return_n_iter: 
            cost_history.append(best_cost)
            
        if clean_cluster >= n_clusters:
            break_iter = i+1
            break

    if verbose:
        print('Final indices:')
        print(medoid_indices)
        print('Final medoids:')
        print(X[medoid_indices])

    # We don't need labels until we're ready to be done
    if precompute_distances:
        labels = np.empty((n_samples,), np.int64)
        distances.argmin(axis=1, out=labels)
    else:
        labels = \
            pairwise_distances_argmin_min(
                X, X[medoid_indices], metric=metric, **metric_kwargs)[0]

    if return_n_iter:
        return medoid_indices, labels, best_cost, break_iter, np.array(cost_history)
    else:
        return medoid_indices, labels, best_cost


class KMedians(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Medians clustering
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        medoids to generate.
    metric : string or callable, default 'euclidean'
        This gets passed to pairwise_distances and determines which metric is
        to be used for the algorithm.
    metric_kwargs : dict
        Keyword arguments for the metric.
    max_iter : int, default: 300
        Maximum number of iterations of the k-medians algorithm for a
        single run.
    tol : float, default: 1e-2
        Tolerance relative to current cost, determines when to stop iteration
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the medoids. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    verbose : int, default 0
        Verbosity mode.
    Attributes
    ----------
    medoids_ : array, (k, n_features)
        Coordinates of medoids
    medoid_indices_ : array, (k,)
        Indices of medoids in training data
    labels_ :
        Labels of each point
    cost_ : float
        Sum of distances of samples to their closest medoid.
    cost_history_ : array, (n_iter+1,)
        History of costs at every iteration
    Notes
    ------
    The k-medians problem is solved using local search.
    """

    def __init__(self, n_clusters=8, metric='euclidean', metric_kwargs=None,
                 init=None, max_iter=300, tol=1e-2, verbose=0,
                 random_state=None, precompute_distances='auto'):

        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.precompute_distances = precompute_distances
        if metric_kwargs is None:
            self.metric_kwargs = dict()
        else:
            self.metric_kwargs = metric_kwargs

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
        expected_n_features = self.medoids_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, y=None, init=None):
        """Compute k-medians clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        init : {None or ndarray or callable}
            Method for initialization, defaults to None:
            None: choose k observations (row indices) at random for
            the initial medoids.
            If an ndarray is passed, it should be of shape (n_clusters,)
            and gives the initial medoid indices within X.
            If a callable is passed, it should accept X, k, and optionally a
            random_state
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        n_samples, n_features = X.shape

        self.medoid_indices_, self.labels_, self.cost_, self.n_iter_, self.cost_history_ = \
            local_search_k_medians(
                X, n_clusters=self.n_clusters, metric=self.metric,
                metric_kwargs=self.metric_kwargs, init=init,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, tol=self.tol, random_state=random_state, precompute_distances=self.precompute_distances)

        self.medoids_ = X[self.medoid_indices_]

        _validate_medoid_shape(X, self.n_clusters, self.medoids_)
        return self

    def fit_predict(self, X, y=None):
        """Compute medoids and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        X = self._check_fit_data(X)
        return self.fit(X)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the medoids.  Note
        that even if X is sparse, the array returned by `transform` will
        typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'medoids_')

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return pairwise_distances(X, self.medoids_, self.metric,
                                  **self.metric_kwargs)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `medoids_` is called
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
        check_is_fitted(self, 'medoids_')

        X = self._check_test_data(X)
        return self._cost(X)[0]

    def score(self, X, y=None):
        """Opposite of the value of X on the K-medians objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        Returns
        -------
        score : float
            Opposite of the value of X on the K-medians objective.
        """
        check_is_fitted(self, 'medoids_')

        X = self._check_test_data(X)
        return -self._cost(X)[1].sum()

    def _cost(self, X):
        return \
            pairwise_distances_argmin_min(
                X, self.medoids_, metric=self.metric, **self.metric_kwargs)
