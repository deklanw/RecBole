r"""
EigenRec
################################################
Reference:
    Athanasios N. Nikolakopoulos et al. "EigenRec: Generalizing PureSVD for Effective and Efficient Top-N Recommendations".

Reference code:
    https://github.com/nikolakopoulos/EigenRec
"""

from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
import scipy.sparse.linalg as linalg

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender

eps = 1e-10

def sparse_cosine_sim(M):
    item_norms = linalg.norm(M, ord=2, axis=0) + eps
    inverse_item_norm_diag = sp.diags(1/item_norms)

    return inverse_item_norm_diag @ M.T @ M @ inverse_item_norm_diag

# https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix
# result has same sparsity as cosine sim but uses dense matrix to speed up computation
# (trading memory for vectorization speed)
def sparse_jaccard_sim(M):
    intersect = (M.T @ M).toarray()
    sums = intersect.diagonal()

    # outer addition
    # trick for vectorized union cardinality
    union = sums[:, None] + sums - intersect + eps
    return sp.csr_matrix(intersect / union)


class EigenRec(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        X = dataset.inter_matrix(
            form='csr').astype(np.float32)

        similarity = config['similarity']
        scaling_factor = config['scaling_factor']
        k = int(config['k'])

        item_norms = linalg.norm(X, ord=2, axis=0) + eps
        S = np.diag(item_norms ** scaling_factor)

        if similarity == 'cosine':
            K = sparse_cosine_sim(X)
        elif similarity == 'jaccard':
            K = sparse_jaccard_sim(X)
        else:
            raise ValueError(
                f"[{similarity}] isn't a valid similarity option.")

        # rescale
        A = S @ K @ S

        _, V = linalg.eigsh(A, k=k)

        item_similarity = V @ V.T

        # torch doesn't support sparse tensor slicing, so will do everything with np/scipy
        self.item_similarity = item_similarity
        self.interaction_matrix = X

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy((self.interaction_matrix[user, :].multiply(self.item_similarity[:, item].T)).sum(axis=1).getA1())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_matrix[user, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())
