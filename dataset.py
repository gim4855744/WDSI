import numpy as np
import torch
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data.dataset import Dataset

from utils import save_pickle, load_pickle


class DataDriver:

    def __init__(self):

        self._numerical_fields_target_names = []
        self._categorical_fields_names = []

        self._minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
        self._label_encoders = {}

        self._num_features_in_categorical_fields = []

    def normalize(self, data, save_path, train=False):

        """
        :param data: dataset (must be a pandas dataframe).
        :param save_path: directory path where the scaler and encoders are/will be saved.
        :param train: if True, the scaler and encoders will be trained.
        :return:
        """

        self._num_features_in_categorical_fields = []  # clear the existing number of features
        if not self._num_features_in_categorical_fields and not self._categorical_fields_names:
            for column in data.columns:
                if data[column].dtype != 'object':
                    self._numerical_fields_target_names.append(column)
                elif data[column].dtype == 'object':
                    self._categorical_fields_names.append(column)
            self._label_encoders = {feature_name: LabelEncoder() for feature_name in self._categorical_fields_names}

        if train:
            self._minmax_scaler.fit(data[self._numerical_fields_target_names])
            for field_name in self._categorical_fields_names:
                self._label_encoders[field_name].fit(data[field_name])
            save_pickle(self._minmax_scaler, os.path.join(save_path, 'minmax_scaler.pkl'))
            save_pickle(self._label_encoders, os.path.join(save_path, 'label_encoders.pkl'))
        else:
            self._minmax_scaler = load_pickle(os.path.join(save_path, 'minmax_scaler.pkl'))
            self._label_encoders = load_pickle(os.path.join(save_path, 'label_encoders.pkl'))

        for field_name in self._categorical_fields_names:
            self._num_features_in_categorical_fields.append(len(self._label_encoders[field_name].classes_))

        numerical_fields_target = self._minmax_scaler.transform(data[self._numerical_fields_target_names])
        numerical_fields = numerical_fields_target[:, :-1]
        target = numerical_fields_target[:, -1]

        categorical_fields = []
        for field_name in self._categorical_fields_names:
            categorical_fields.append(self._label_encoders[field_name].transform(data[field_name]))

        numerical_fields = np.array(numerical_fields)
        categorical_fields = np.array(categorical_fields).transpose()
        target = np.array(target).reshape(-1, 1)

        return numerical_fields, categorical_fields, target

    def get_num_numerical_fields(self):
        # without target
        return len(self._numerical_fields_target_names) - 1

    def get_num_categorical_fields(self):
        return len(self._categorical_fields_names)

    def get_num_features_in_categorical_fields(self):
        return self._num_features_in_categorical_fields


class TensorDataset(Dataset):

    def __init__(self, numerical_fields, categorical_fields, target):

        self._numerical_fields = torch.tensor(numerical_fields, dtype=torch.float32)
        self._categorical_fields = torch.tensor(categorical_fields, dtype=torch.int32)
        self._target = torch.tensor(target, dtype=torch.float32)
        self._num_data = self._target.shape[0]

    def __getitem__(self, index):
        return self._numerical_fields[index], self._categorical_fields[index], self._target[index]

    def __len__(self):
        return self._num_data
