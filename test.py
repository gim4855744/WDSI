import math

import pandas as pd
import torch
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data.dataloader import DataLoader
from absl import app, flags

from dataset import DataDriver, TensorDataset
from model import WDSI

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'dataset directory path')
flags.DEFINE_string('save_dir', '', 'save directory path')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(argv):

    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    batch_size = 128
    emb_size = 8

    train_data_path = os.path.join(data_dir, 'train.csv')

    test_data = pd.read_csv(train_data_path)
    data_driver = DataDriver()

    test_numerical_fields, test_categorical_fields, test_target = \
        data_driver.normalize(test_data, save_dir, train=False)

    test_dataset = TensorDataset(test_numerical_fields, test_categorical_fields, test_target, device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_numerical_fields = data_driver.get_num_numerical_fields()
    num_categorical_fields = data_driver.get_num_categorical_fields()
    num_features_in_categorical_fields = data_driver.get_num_features_in_categorical_fields()

    model = WDSI(num_numerical_fields, num_categorical_fields, num_features_in_categorical_fields, emb_size).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'WDSI.pth')))
    model.eval()

    total_predicts = []

    with torch.no_grad():
        for batch_numerical_fields, batch_categorical_fields, _ in test_loader:
            predicts = model(batch_numerical_fields, batch_categorical_fields)
            total_predicts.extend(predicts.detach().cpu().tolist())

    rmse = math.sqrt(mean_squared_error(test_target, total_predicts))
    mae = mean_absolute_error(test_target, total_predicts)
    r2 = r2_score(test_target, total_predicts)

    print("RMSE: {}, MAE: {}, R^2: {}".format(rmse, mae, r2))


if __name__ == '__main__':
    app.run(main)
