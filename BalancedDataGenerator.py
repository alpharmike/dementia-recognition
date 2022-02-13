import numpy as np
import keras
from sklearn.utils.multiclass import unique_labels


class BalancedDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, x, y, batch_size=32, shuffle=True):
        'Initialization'

        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.count = self.__get_counts()  # count of data in each class
        self.balanced_count = self.__get_balanced_counts()  # the right combination for a balanced mini-batch
        self.Ids = self.__get_Ids()  # get index column of dataframe
        self.shuffle = shuffle
        self.on_epoch_end()

    def __get_counts(self):
        labels = self.y.iloc[:, 0]
        return labels.value_counts()

    def __get_balanced_counts(self):
        ratios = self.count / self.count.sum()
        bc = np.round(self.batch_size * ratios)
        for c in bc.index:
            if bc[c] == 0:
                bc[c] = 1
                if bc.sum() > self.batch_size:
                    mx = max(bc)
                    bc[bc == mx] = mx - 1

        return bc

    def __get_Ids(self):
        classes = self.count.index
        ids = {}
        labels = self.y.iloc[:, 0]
        for c in classes:
            cids = labels[labels == c].index.values
            ids[c] = cids

        return ids

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = np.floor((self.count / self.balanced_count.values).values)
        return int(min(num_batches))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_Ids = []
        bc = self.balanced_count
        for c in bc.index:
            cids = self.Ids[c]
            start = int(index * bc[c])
            end = int((index + 1) * bc[c])
            batch_cids = list(cids[start:end])
            batch_Ids = batch_Ids + batch_cids

        # Generate data
        X, y = self.__data_generation(batch_Ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        classes = self.count.index
        if self.shuffle == True:
            for c in classes:
                np.random.shuffle(self.Ids[c])

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Generate data
        X = self.x.loc[list_IDs_temp, :].values.astype(float)
        y = self.y.loc[list_IDs_temp, :].values

        return X, y
