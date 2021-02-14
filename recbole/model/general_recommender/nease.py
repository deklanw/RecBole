r"""
NEASE
################################################
Reference:
"""


from recbole.model.loss import BPRLoss
from recbole.utils.enum_type import ModelType
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


def scipy_to_sparse_tensor(A):
    # https://stackoverflow.com/a/50665264/7367514
    C = A.tocoo()

    values = C.data
    indices = np.vstack((C.row, C.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = C.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class NEASE(GeneralRecommender):
    input_type = InputType.PAIRWISE
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(
            form='csr').astype(np.float32)

        _, num_items = self.interaction_matrix.shape

        bias = config['bias']
        hide_diagonals = config['hide_diagonals']

        self.linear = nn.Linear(num_items, num_items, bias=bias)
        self.bpr_loss = BPRLoss()

        if hide_diagonals:
            # backward hook to prevent diagonal from changing
            def hook_fn(grad):
                # You are not allowed to modify inplace what is given
                out = grad.clone()
                out.fill_diagonal_(0)
                return out

            # make sure diagonals are initially 0
            with torch.no_grad():
                self.linear.weight.fill_diagonal_(0)

            self.linear.weight.register_hook(hook_fn)

    def forward(self, users):
        # can't slice PT tensor
        user_interactions = scipy_to_sparse_tensor(
            self.interaction_matrix[users]).to(self.device)

        return self.linear(user_interactions)

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_predictions = self.forward(users)

        # https://stackoverflow.com/questions/61311688/index-a-torch-tensor-with-an-array
        pos_scores = user_predictions.gather(1, pos_item.unsqueeze(1))
        neg_scores = user_predictions.gather(1, neg_item.unsqueeze(1))

        return self.bpr_loss(pos_scores, neg_scores)

    def predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        pos_item = interaction[self.ITEM_ID]

        user_predictions = self.forward(users)
        pos_scores = user_predictions.gather(1, pos_item.unsqueeze(1))

        return pos_scores

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()

        user_predictions = self.forward(users)

        return torch.flatten(user_predictions)
