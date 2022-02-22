import torch

from torch.nn import Module, Parameter
from torch.nn import Embedding

from component import WideComponent, DeepComponent


class WDSI(Module):

    def __init__(self, num_numerical_fields, num_categorical_fields, num_features_in_categorical_fields, emb_size):

        super(WDSI, self).__init__()

        self._embeddings = [Embedding(num_features, emb_size) for num_features in num_features_in_categorical_fields]

        num_features = num_numerical_fields + num_categorical_fields * emb_size

        wide_architecture = ((num_features, 1000),
                             (1000, 1))
        self._wide = WideComponent(wide_architecture)

        deep_architecture = ((num_features, 128),
                             (128, 64),
                             (64, 32),
                             (32, 1))
        self._deep = DeepComponent(deep_architecture)

        self._bias = Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self._bias)

    def forward(self, numerical_fields, categorical_fields):

        features = [numerical_fields]
        for i, embedding in enumerate(self._embeddings):
            features.append(embedding(categorical_fields[:, i]))
        features = torch.cat(features, dim=1)

        predicts = self._wide(features) + self._deep(features) + self._bias

        return predicts
