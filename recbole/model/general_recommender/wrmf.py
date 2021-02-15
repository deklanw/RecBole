r"""
WRMF
################################################
Reference:
"""

from recbole.utils.enum_type import ModelType
import numpy as np
import torch

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender

import implicit


class WRMF(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.interaction_matrix = dataset.inter_matrix(
            form='csr').astype(np.float32)

        embedding_size = config['embedding_size']
        reg_weight = config['reg_weight']
        iterations = config['iterations']
        seed = config['seed']

        self.model = implicit.als.AlternatingLeastSquares(
            factors=embedding_size, regularization=reg_weight, iterations=iterations, random_state=seed)

        self.model.fit(self.interaction_matrix.T)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        items = interaction[self.ITEM_ID].cpu().numpy()

        recs = []
        for user, item in zip(users, items):
            user_factors = self.model.user_factors[user]
            score = self.model.item_factors[item].dot(user_factors)
            recs.append(score)

        return torch.from_numpy(np.array(recs))

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()

        recs = []
        for user in users:
            user_factors = self.model.user_factors[user]
            scores = self.model.item_factors.dot(user_factors)
            recs.append(scores)

        return torch.from_numpy(np.vstack(recs).flatten())
