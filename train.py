import pandas as pd
import numpy as np
import random
import torch
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from absl import app, flags

from model import WDSI

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/home/minkyu/Datasets/HousePrices', 'dataset directory path')  # the default value will be removed
flags.DEFINE_string('save_dir', './save/', 'save directory path')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(argv):

    data_dir = FLAGS.data_dir
    val_size = 0.1
    emb_size = 8

    train_data_path = os.path.join(data_dir, 'train.csv')

    data = pd.read_csv(train_data_path)
    train_data, val_data = train_test_split(data, test_size=val_size)

    numerical_fields_target_names = [column for column in data.columns if data[column].dtype != 'object']
    categorical_fields_names = [column for column in data.columns if data[column].dtype == 'object']

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_numerical_fields_target = minmax_scaler.fit_transform(train_data[numerical_fields_target_names])
    val_numerical_fields_target = minmax_scaler.transform(val_data[numerical_fields_target_names])
    train_numerical_fields = train_numerical_fields_target[:, :-1]
    val_numerical_fields = val_numerical_fields_target[:, :-1]
    train_target = train_numerical_fields_target[:, -1]
    val_target = val_numerical_fields_target[:, -1]
    # save scaler

    label_encoders = {feature_name: LabelEncoder() for feature_name in categorical_fields_names}
    train_categorical_fields, val_categorical_fields = [], []
    num_features_in_categorical_fields = []
    for field_name in categorical_fields_names:
        train_categorical_fields.append(label_encoders[field_name].fit_transform(train_data[field_name]))
        val_categorical_fields.append(label_encoders[field_name].transform(val_data[field_name]))
        num_features_in_categorical_fields.append(len(label_encoders[field_name].classes_))
    # save encoders

    train_numerical_fields = np.array(train_numerical_fields)
    train_categorical_fields = np.array(train_categorical_fields).transpose()
    train_target = np.array(train_target).reshape(-1, 1)

    val_numerical_fields = np.array(val_numerical_fields)
    val_categorical_fields = np.array(val_categorical_fields).transpose()
    val_target = np.array(val_target).reshape(-1, 1)

    train_numerical_fields = torch.tensor(train_numerical_fields, dtype=torch.float32)
    train_categorical_fields = torch.tensor(train_categorical_fields, dtype=torch.int32)
    train_target = torch.tensor(train_target, dtype=torch.float32)

    num_numerical_fields = train_numerical_fields.shape[1]
    num_categorical_fields = train_categorical_fields.shape[1]

    model = WDSI(num_numerical_fields, num_categorical_fields, num_features_in_categorical_fields, emb_size)
    predicts = model(train_numerical_fields, train_categorical_fields)
    print(predicts.shape)


if __name__ == '__main__':
    app.run(main)
