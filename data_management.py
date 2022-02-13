from typing import List
import pandas as pd
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 100)
np.set_printoptions(linewidth=desired_width, threshold=sys.maxsize)


def split_data(data_file: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    # load data
    df = pd.read_csv(data_file, index_col=0)
    print('original: ', df.shape)

    df = df.dropna()  # remove nulls
    print('after dropna: ', df.shape)

    # split datasets into input train, validation & test subsets
    y_val, rest = get_subset(df, 1, 1)
    y_test, y_train = get_subset(rest, 1, 0)

    # save data
    y_train.to_csv(os.path.join(save_path, 'y_train.csv'))
    y_val.to_csv(os.path.join(save_path, 'y_val.csv'))
    y_test.to_csv(os.path.join(save_path, 'y_test.csv'))


def get_subset(df: pd.DataFrame, n: int, keep: int):
    # remove response variables
    # resp_vars = ['DWM', 'PVWM', 'GCA']
    y_values = [0, 1, 2, 3]

    indices = []
    for dwm in y_values:
        for pvwm in y_values:
            for gca in y_values:
                rows = df[(df['DWM'] == dwm) & (df['PVWM'] == pvwm) & (df['GCA'] == gca)]
                if len(rows) > (n + keep):
                    sel = np.random.choice(rows.index, n).tolist()
                    indices = indices + sel

    subset = df.loc[indices, :]
    rest = df.drop(indices, axis=0)
    return subset, rest


def load_data(data_file: str, values: bool = True):
    # load data
    df = pd.read_csv(data_file, index_col=0)
    print('data shape: ', df.shape)

    # convert to float
    if values:
        return df.values.astype(float)
    else:
        return df


def get_info(data_path: str):
    train_file = os.path.join('', 'y_train.csv')
    val_file = os.path.join('', 'y_val.csv')
    test_file = os.path.join('', 'y_test.csv')

    y_train = load_data(train_file, values=False)
    y_val = load_data(val_file, values=False)
    y_test = load_data(test_file, values=False)

    resp_vars = ['DWM', 'PVWM', 'GCA']
    train_count = pd.DataFrame()
    val_count = pd.DataFrame()
    test_count = pd.DataFrame()
    for var in resp_vars:
        train_count = pd.concat([train_count, y_train[var].value_counts()], axis=1)
        val_count = pd.concat([val_count, y_val[var].value_counts()], axis=1)
        test_count = pd.concat([test_count, y_test[var].value_counts()], axis=1)

    print('----- > train < -----\n', train_count)
    print('----- >  val  < -----\n', val_count)
    print('----- >  test < -----\n', test_count)


if __name__ == '__main__':
    root = ''

    data_file = os.path.join('', 'AIMIN_Challenge_Training_Labels.csv')
    split_data_path = os.path.join(root, 'Data/AIMIN_Challenge_Training_Dataset/Selected_Data/')

    ## split data
    split_data(data_file, split_data_path)
    get_info(split_data_path)
