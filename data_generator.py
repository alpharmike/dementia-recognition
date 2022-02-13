import os
import pydicom as dicom
import imageio
import cv2
import numpy as np
from keras.utils import Sequence, to_categorical
from PIL import Image


class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels, batch_size=32, dim=(256, 256), n_channels=24,
                 n_classes=4, shuffle=True):
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.generate_data(self.path, list_IDs_temp)

        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID)

            # Store class
            y[i] = self.labels[ID]
        return X, to_categorical(y, num_classes=self.n_classes)

    def generate_data(self, base_path, list_id):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 3), dtype=int)
        for index, id in enumerate(list_id):
            img_file_path = base_path + '/' + str(id)
            chn = 0
            for img_file in os.listdir(img_file_path):
                img_path = img_file_path + '/' + img_file
                image = imageio.imread(img_path)
                image = image.astype('float32')
                res_image = cv2.resize(image, self.dim)
                X[index, :, :, chn] = res_image
                chn = chn + 1

            # image_file = np.reshape(image_file, )
            y[index] = self.labels.loc[id]  # labels as dictionary
        return X, [to_categorical(y[:, 0], self.n_classes),
                   to_categorical(y[:, 1], self.n_classes),
                   to_categorical(y[:, 2], self.n_classes)]
