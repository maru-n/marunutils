# -*- coding: utf-8 -*-

import numpy as np
import warnings
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.utils.validation import _check_sample_weight
from sklearn.neighbors import NearestNeighbors

# modify DBSCAN to save and reuse neighbors data.
# Original code
# https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/cluster/_dbscan.py#L148

class DBSCAN(DBSCAN):

    def fit(self, X, y=None, sample_weight=None, neighbors=None):
        X = self._validate_data(X, accept_sparse='csr')

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        neighborhoods = neighbors
        if neighborhoods is None:
            neighbors_model = NearestNeighbors(
                radius=self.eps, algorithm=self.algorithm,
                leaf_size=self.leaf_size, metric=self.metric,
                metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
            neighbors_model.fit(X)
            # This has worst case O(n^2) memory complexity
            neighborhoods = neighbors_model.radius_neighbors(X,
                                                            return_distance=False)
        self.neighbors_ = neighborhoods

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self


    def fit_predict(self, X, y=None, sample_weight=None, neighbors=None):
        self.fit(X, sample_weight=sample_weight, neighbors=neighbors)
        return self.labels_
