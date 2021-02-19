r"""
"""


from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
import scipy
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
        self.depth = config['depth']
        num_hashes = config['num_hashes']
        embedding_type = config['embedding_type']
        prior_weight = config['prior_weight']

        num_buckets = 2**num_hashes

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

        item_sketches = emde(item_embeddings, self.depth, num_hashes, d)

        user_prefs = X @ (item_sketches + prior_weight)

        # normalize each of the [depth] number of probability distributions
        user_prefs /= user_prefs[:, :num_buckets].sum(axis=1).reshape(-1, 1)

        self.user_prefs = user_prefs
        self.nonzero_item_sketch_columns = item_sketches.nonzero()[1]

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def get_all_user_predictions(self, user):
        # little tricky here
        # for each item:
        # get the probability for the user according to each of the self.depth probability distributions
        user_all_items_full_depth = self.user_prefs[user][self.nonzero_item_sketch_columns].reshape(
            -1, self.depth)

        # then take the geometric mean
        return scipy.stats.mstats.gmean(user_all_items_full_depth, axis=1)

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        predictions = []
        for u, i in zip(user, item):
            a = self.get_all_user_predictions(u)[i]
            predictions.append(a)

        return torch.from_numpy(np.array(predictions))

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        predictions = []
        for u in user:
            ps = self.get_all_user_predictions(u)
            predictions.append(ps)

        return torch.from_numpy(np.concatenate(predictions))
