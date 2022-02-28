import pandas as pd
import numpy as np
import random
import torch
import os

from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from absl import app, flags

from model import WDSI
from dataset import DataDriver, TensorDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'dataset directory path')
flags.DEFINE_string('save_dir', '', 'save directory path')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(argv):

    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    val_size = 0.1
    emb_size = 8
    batch_size = 128
    lr = 1e-4
    epochs = 100

    train_data_path = os.path.join(data_dir, 'train.csv')

    data = pd.read_csv(train_data_path)
    data_driver = DataDriver()
    train_data, val_data = train_test_split(data, test_size=val_size)

    train_numerical_fields, train_categorical_fields, train_target =\
        data_driver.normalize(train_data, save_dir, train=True)
    val_numerical_fields, val_categorical_fields, val_target =\
        data_driver.normalize(val_data, save_dir, train=False)

    train_dataset = TensorDataset(train_numerical_fields, train_categorical_fields, train_target, device)
    val_dataset = TensorDataset(val_numerical_fields, val_categorical_fields, val_target, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_numerical_fields = data_driver.get_num_numerical_fields()
    num_categorical_fields = data_driver.get_num_categorical_fields()
    num_features_in_categorical_fields = data_driver.get_num_features_in_categorical_fields()

    model = WDSI(num_numerical_fields, num_categorical_fields, num_features_in_categorical_fields, emb_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    min_val_loss = 9999.

    for epoch in range(epochs):

        train_losses, val_losses = [], []

        model.train()
        for batch_numerical_fields, batch_categorical_fields, batch_target in train_loader:
            optimizer.zero_grad()
            predicts = model(batch_numerical_fields, batch_categorical_fields)
            train_loss = criterion(predicts, batch_target)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

        with torch.no_grad():
            model.eval()
            for batch_numerical_fields, batch_categorical_fields, batch_target in val_loader:
                predicts = model(batch_numerical_fields, batch_categorical_fields)
                val_loss = criterion(predicts, batch_target)
                val_losses.append(val_loss.item())

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)

        print("Epoch {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(epoch + 1,
                                                                          mean_train_loss,
                                                                          mean_val_loss))

        if min_val_loss > mean_val_loss:
            min_val_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'WDSI.pth'))

    print("Best Val Loss: {}".format(min_val_loss))


if __name__ == '__main__':
    app.run(main)
