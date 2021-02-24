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


def local_slim_similarity(M, alpha, l1_ratio, C):
    _, num_items = M.shape

    # get item-item cosine similarity
    S = cosine_similarity(M.T, dense_output=True)

    # remove self-similarity
    np.fill_diagonal(S, 0)

    # find top C NN for each item
    top_C_neighbors = np.argsort(S)[:, -C:]

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

    # make column slicing more efficient
    M = M.tocsc()

    # solve the SLIM problem for each item and its C neighbors, and store coefficients in appropriate column of W
    # ignore ConvergenceWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        coeffs = Parallel(n_jobs=-1)(delayed(fit_elastic)(M[:, top_C_neighbors[i]], M[:, i].A)
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


def scipy_to_sparse_tensor(A):
    # https://stackoverflow.com/a/50665264/7367514
    C = A.tocoo()

    values = C.data
    indices = np.vstack((C.row, C.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = C.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


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
        self.k = int(config['k'])
        self.damping_factor = config['damping_factor']
        self.mode = config['mode']
        self.tol = 1e-6

        P = compute_P(B, alpha, l1_ratio, C, cross_probability)

        # store this transposed
        self.P_t = scipy_to_sparse_tensor(P.T).to(self.device)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def get_walk_indicator(self, users):
        batch_size = len(users)
        num_users, num_items = self.shape

        walk_indicators = torch.zeros((batch_size, num_users + num_items)).to(self.device)

        walk_indicators[range(len(users)), users] = 1

        return walk_indicators

    def get_user_predictions_k_step(self, users):
        num_users, _ = self.shape

        walk_indicators = self.get_walk_indicator(users)
        current_scores_t = walk_indicators.clone().transpose(0, 1)

        for _ in range(self.k):
            current_scores_t = torch.sparse.mm(self.P_t, current_scores_t)

        # make all item predictions for specified users
        user_all_items = current_scores_t.transpose(0, 1)[:, num_users:]

        return user_all_items

    def get_user_predictions_pagerank(self, users):
        num_users, _ = self.shape

        walk_indicators_t = self.get_walk_indicator(users).transpose(0, 1)
        last = walk_indicators_t

        # PR power method until convergence
        while True:
            current_scores = (self.damping_factor * torch.sparse.mm(self.P_t, last)
                              + (1-self.damping_factor) * walk_indicators_t)

            current_scores = current_scores / \
                torch.linalg.norm(current_scores, ord=2, dim=0)

            residuals = torch.linalg.norm(current_scores - last, ord=1, dim=0)

            if torch.all(residuals < self.tol):
                break

            last = current_scores

        # make all item predictions for specified users
        user_all_items = current_scores.transpose(0, 1)[:, num_users:]

        return user_all_items

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]

        if self.mode == 'kstep':
            user_all_items = self.get_user_predictions_k_step(users)
        else:
            user_all_items = self.get_user_predictions_pagerank(users)

        # then narrow down to specific items
        item_predictions = user_all_items[range(len(users)), items]

        return item_predictions.flatten()

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]

        if self.mode == 'kstep':
            user_all_items = self.get_user_predictions_k_step(users)
        else:
            user_all_items = self.get_user_predictions_pagerank(users)

        return user_all_items.flatten()
