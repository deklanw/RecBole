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
import scipy

from scipy.sparse import linalg
from sklearn.preprocessing import normalize

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

def random_walk_proj(M):
    row_normalized = normalize(M, norm='l1', axis=1)
    return row_normalized.T @ row_normalized

def cosine_proj(M):
    item_norms = linalg.norm(M, ord=2, axis=0) + eps
    inverse_item_norm_diag = sp.diags(1/item_norms)
    
    return inverse_item_norm_diag @ M.T @ M @ inverse_item_norm_diag

def hyperbolic_proj(M):
    user_interaction_nums = M.sum(axis=1).getA1() + eps
    inverse_user_interaction_nums = sp.diags(1/user_interaction_nums)
    
    return M.T @ inverse_user_interaction_nums @ M 

def partial_configuration_proj(M, p_val=0.05):
    # BiPCM
    # Inferring monopartite projections of bipartite networks: an entropy-based approach
    item_degrees = M.sum(axis=0).getA1()
    
    num_users, num_items = M.shape
    square_users = num_users ** 2
    
    projection = sp.lil_matrix((num_items, num_items))
    length_3_paths = sp.triu(M.T @ M).tocsr()
    
    surprise_cutoff = -np.log(p_val)
    
    for (i, j) in zip(*length_3_paths.nonzero()):
        observed = length_3_paths[i, j]
        
        p = item_degrees[i] * item_degrees[j] / square_users
        surprise = -scipy.stats.binom.logsf(observed, num_users, p)
        
        if surprise > surprise_cutoff:
            projection[i, j] = surprise
            projection[j, i] = surprise

    return projection


class RecWalk(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        B = dataset.inter_matrix(
            form='coo').astype(np.float32)

        alpha = config['alpha']
        similarity_model = config['similarity_model']
        k = int(config['k'])

        if similarity_model == 'rw':
            W = random_walk_proj(B)
        elif similarity_model == 'pcm':
            W = partial_configuration_proj(B)
        elif similarity_model == 'cos':
            W = cosine_proj(B)
        elif similarity_model == 'hyp':
            W = hyperbolic_proj(B)
        else:
            raise ValueError("Invalid similarity model")

        A = adjacency_of_bipartite(B)
        H = sp.diags(1 / (A.sum(axis=1) + eps).getA1()) @ A

        W_inf_norm = linalg.norm(W, ord=np.inf)
        M_I = W / W_inf_norm + sp.diags(1 - W.sum(axis=1).getA1() / W_inf_norm)
        M = construct_M(M_I, B.shape)

        P = alpha * H + (1-alpha) * M
        self.walk = P ** k
        self.shape = B.shape


    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        batch_size = len(user)
        num_users, num_items = self.shape

        walk_indicators = sp.lil_matrix((batch_size, num_users + num_items))

        for i, user_id in enumerate(user):
            walk_indicators[i, user_id] = 1
        
        # make all item predictions for specified users
        user_all_items = (walk_indicators @ self.walk)[:, num_users:]

        # then narrow down to specific items
        # without this copy(): "cannot set WRITEABLE flag..."
        item_predictions = user_all_items[range(len(user)), item.copy()]

        return torch.from_numpy(item_predictions.getA1())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        batch_size = len(user)
        num_users, num_items = self.shape

        walk_indicators = sp.lil_matrix((batch_size, num_users + num_items))

        for i, user_id in enumerate(user):
            walk_indicators[i, user_id] = 1
        
        item_predictions = (walk_indicators @ self.walk)[:, num_users:]

        return torch.from_numpy(item_predictions.todense().getA1())