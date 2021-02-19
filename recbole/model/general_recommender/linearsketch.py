r"""
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
import warnings
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


def adjacency_of_bipartite(B):
    m, n = B.shape

    Z1 = sp.coo_matrix((m, m))
    Z2 = sp.coo_matrix((n, n))

    A = sp.bmat([[Z1, B], [B.T, Z2]])

    return A


def simple_bipartite_smoothing(initial, R, iters=4):
    W = adjacency_of_bipartite(R)
    T = initial

    for _ in range(iters):
        T = W @ T
        T = normalize(T, axis=1, norm='l2')

    return T


def simple_bipartite_embedding(R, d):
    num_users, num_items = R.shape
    N = num_users + num_items

    # initialize item embeddings randomly
    # rescale and shift to be uniform from -1 to 1
    initial = 2 * np.random.rand(N, d) - 1

    simple_embedding = simple_bipartite_smoothing(initial, R)

    return simple_embedding


def randomized_svd_embedding(R, d):
    # now get the SVD embedding
    U, sigma, Vt = randomized_svd(R, n_components=d,
                                  n_iter='auto',
                                  power_iteration_normalizer='QR')

    sqrt_Sigma = np.diag(np.power(sigma, 1/2))

    svd_user_embeddings = U @ sqrt_Sigma
    svd_item_embeddings = Vt.T @ sqrt_Sigma
    svd_embeddings = np.vstack([svd_user_embeddings, svd_item_embeddings])

    return svd_embeddings


def smoothed_svd(R, d):
    return simple_bipartite_smoothing(randomized_svd_embedding(R, d), R)


'''
Fast way is vectorized matmult. Pass in all rows and cols in one shot.

https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
'''


def bits_to_ints(bits):
    n = bits.shape[1]
    a = 2**np.arange(n)[::-1]
    return bits @ a


def emde(item_embeddings, depth, num_hashes, d):
    # might be able to vectorize all of this, but it would probably be more confusing
    # already negligibly fast

    num_items = item_embeddings.shape[0]

    sketch_rows = []

    for _ in range(depth):
        # create a random vector in the embedding space for every hash
        r = np.random.rand(d, num_hashes)

        num_buckets = 2**num_hashes

        dots = item_embeddings @ r
        random_quantiles = np.random.rand(num_hashes)
        biases = np.array([np.quantile(dots[:, i], random_quantiles[i])
                           for i in range(num_hashes)])
        bits = np.array(dots < biases, dtype=np.int32)
        ints = bits_to_ints(bits)

        sketch_row = np.zeros((num_items, num_buckets))
        sketch_row[range(num_items), ints] = 1
        sketch_rows.append(sketch_row)

    item_sketches = np.hstack(sketch_rows)

    return item_sketches


class LinearSketch(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        d = config['embedding_size']
        alpha = config['alpha']
        embedding_type = config['embedding_type']
        depth = config['depth']
        num_hashes = config['num_hashes']

        X = dataset.inter_matrix(
            form='csr').astype(np.float32)

        num_users, _ = X.shape

        if embedding_type == 'simple':
            embeddings = simple_bipartite_embedding(X, d)
        elif embedding_type == 'svd':
            embeddings = randomized_svd_embedding(X, d)
        elif embedding_type == 'smoothed_svd':
            embeddings = smoothed_svd(X, d)

        item_embeddings = embeddings[num_users:]

        item_sketches = emde(item_embeddings, depth, num_hashes, d)
        self.user_sketches = (X @ item_sketches) / (X.sum(axis=1) + 1e-7)

        model = Ridge(alpha=alpha)

        coeffs = []

        # ignore ConvergenceWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for j in range(X.shape[1]):
                # target column
                r = X[:, j].todense().getA1()

                # fit the model
                model.fit(self.user_sketches, r)

                # store the coefficients
                cs = model.coef_

                coeffs.append(cs)

        self.coefficients = np.vstack(coeffs).T

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy((self.user_sketches[user, :] * self.coefficients[:, item].T).sum(axis=1).flatten())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.user_sketches[user, :] @ self.coefficients
        return torch.from_numpy(r.flatten())
