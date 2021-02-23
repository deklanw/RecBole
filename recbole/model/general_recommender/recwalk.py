r"""
RecWalk
################################################
Reference:

Reference code:
"""

from recbole.utils.enum_type import ModelType
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender

import torch
import numpy as np
import scipy.sparse as sp
import warnings

from scipy.sparse import linalg
from sklearn.preprocessing import normalize
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import ConvergenceWarning


from joblib import Parallel, delayed


eps = 1e-6


def adjacency_of_bipartite(B):
    m, n = B.shape

    Z1 = sp.coo_matrix((m, m))
    Z2 = sp.coo_matrix((n, n))

    A = sp.bmat([[Z1, B], [B.T, Z2]])

    return A


def construct_M(M_I, shape):
    m, n = shape

    I = sp.identity(m, format="coo")
    Z1 = sp.coo_matrix((m, n))
    Z2 = sp.coo_matrix((n, m))

    A = sp.bmat([[I, Z1], [Z2, M_I]])

    return A


def top_n_idx_sparse(matrix, n):
    # https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
    '''Return index of (up to) top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(
            matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


def local_slim_similarity(M, alpha, l1_ratio, C):
    _, num_items = M.shape

    # get item-item cosine similarity
    S = cosine_similarity(M.T, dense_output=False).tolil()

    # remove self-similarity
    S.setdiag(0)

    S = S.tocsr()

    # find top C NN for each item
    top_C_neighbors = top_n_idx_sparse(S, C)

    def fit_elastic(X, y):
        # if no neighbors. edge case where item has no interactions
        if X.shape[1] == 0:
            return []

        # setup ElasticNet for solving local SLIM for each item
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           positive=True,
                           fit_intercept=False,
                           copy_X=True,
                           precompute=True,
                           selection='random',
                           max_iter=500,
                           tol=1e-4)

        model.fit(X, y)

        return model.coef_

    # solve the SLIM problem for each item and its C neighbors, and store coefficients in appropriate column of W
    # ignore ConvergenceWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        coeffs = Parallel(n_jobs=-1)(delayed(fit_elastic)(M[:, top_C_neighbors[i]], M[:, i].todense().getA1())
                                     for i in range(num_items))

    # empty item-item similarity matrix
    W = sp.lil_matrix((num_items, num_items))

    # might be a smarter way to construct this sparse matrix, but this is fine.
    # negligible cost compared to fitting
    for i, (ns, cs) in enumerate(zip(top_C_neighbors, coeffs)):
        W[ns, i] = cs

    return W


def compute_P(B, alpha, l1_ratio, C, cross_probability):
    W = local_slim_similarity(B, alpha, l1_ratio, C)

    A = adjacency_of_bipartite(B)
    H = sp.diags(1 / (A.sum(axis=1) + eps).getA1()) @ A

    W_inf_norm = linalg.norm(W, ord=np.inf)
    M_I = W / W_inf_norm + sp.diags(1 - W.sum(axis=1).getA1() / W_inf_norm)
    M = construct_M(M_I, B.shape)

    P = cross_probability * H + (1-cross_probability) * M

    return P


class RecWalk(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        B = dataset.inter_matrix(
            form='csr').astype(np.float32)

        self.shape = B.shape

        alpha = config['alpha']
        l1_ratio = config['l1_ratio']
        cross_probability = config['cross_probability']
        C = int(config['C'])
        k = int(config['k'])
        self.damping_factor = config['damping_factor']
        self.mode = config['mode']
        self.tol = 1e-6

        self.P = compute_P(B, alpha, l1_ratio, C, cross_probability)

        if self.mode == 'kstep':
            # precompute this for predictions
            # could consider saving memory by avoiding this
            self.walk = self.P.todense() ** k
        elif self.mode == 'pagerank':
            pass
        else:
            raise ValueError("Mode should be 'kstep' or 'pagerank'")

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def get_walk_indicator(self, users):
        batch_size = len(users)
        num_users, num_items = self.shape

        walk_indicators = sp.lil_matrix((batch_size, num_users + num_items))

        for i, user_id in enumerate(users):
            walk_indicators[i, user_id] = 1

        return walk_indicators

    def get_user_predictions_k_step(self, users):
        num_users, num_items = self.shape

        walk_indicators = self.get_walk_indicator(users)

        # make all item predictions for specified users
        user_all_items = (walk_indicators @ self.walk)[:, num_users:]

        return user_all_items

    def get_user_predictions_pagerank(self, users):
        num_users, _ = self.shape

        walk_indicators = self.get_walk_indicator(users)
        last = walk_indicators.copy().todense()

        # PR power method until convergence
        while True:
            current_scores = (self.damping_factor *
                              (last @ self.P) + (1-self.damping_factor) * walk_indicators)

            current_scores = normalize(current_scores, norm='l2', axis=1)

            residuals = np.linalg.norm(current_scores - last, ord=1, axis=1)

            if np.all(residuals < self.tol):
                break

            last = current_scores

        # make all item predictions for specified users
        user_all_items = current_scores[:, num_users:]

        return user_all_items

    def predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        items = interaction[self.ITEM_ID].cpu().numpy()

        if self.mode == 'kstep':
            user_all_items = self.get_user_predictions_k_step(users)
        else:
            user_all_items = self.get_user_predictions_pagerank(users)

        # then narrow down to specific items
        # without this copy(): "cannot set WRITEABLE flag..."
        item_predictions = user_all_items[range(len(users)), items.copy()]

        return torch.from_numpy(item_predictions.flatten())

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()

        if self.mode == 'kstep':
            user_all_items = self.get_user_predictions_k_step(users)
        else:
            user_all_items = self.get_user_predictions_pagerank(users)

        return torch.from_numpy(user_all_items.flatten())
