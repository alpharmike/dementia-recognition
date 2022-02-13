from typing import List
import pandas as pd
import numpy as np
import sys
import os
from Models import get_model
from BalancedDataGenerator import BalancedDataGenerator
from Utils import weighted_categorical_crossentropy
import json
from data_management import load_data
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import save_model, load_model
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Model
from keras.layers import Activation, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.regularizers import l1, l1_l2, l2
from data_generator import DataGenerator

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 100)
np.set_printoptions(linewidth=desired_width, threshold=sys.maxsize)
csv_read_train = pd.read_csv('y_train.csv')
csv_read_val = pd.read_csv('y_val.csv')
list_IDs_train = csv_read_train['Study ID'].tolist()
list_IDs_val = csv_read_val['Study ID'].tolist()


def train_model(root: str, data_path: str, model_parameters: List[str]):
    # create directories

    model_name = '-'.join(model_parameters)
    weight_dir = os.path.join(root, 'NN/weights/%s/' % model_name)
    log_dir = os.path.join(root, 'NN/logs/%s/' % model_name)
    models_dir = os.path.join(root, 'NN/Models/')

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # load data
    train_file = os.path.join(data_path, 'y_train.csv')
    val_file = os.path.join(data_path, 'y_val.csv')

    y_train = load_data(train_file, values=False)
    y_val = load_data(val_file, values=False)

    # build model
    nn_model = build_model('unet', (256, 256, 24))

    # calculate class weights
    weights = {}
    resp_vars = ['DWM', 'PVWM', 'GCA']

    for var in resp_vars:
        y_val = y_train[var].values
        weights[var] = class_weight.compute_class_weight('balanced', np.unique(y_val), y_val.flatten())

    print('class weights: ', weights)

    # define custom weighted loss for each output
    losses = {
        "DWM_output": weighted_categorical_crossentropy(np.float32(weights['DWM']), "DWM_loss"),

        "PVWM_output": weighted_categorical_crossentropy(np.float32(weights['PVWM']), "PVWM_loss"),
        "GCA_output": weighted_categorical_crossentropy(np.float32(weights['GCA']), "GCA_loss")
    }

    # set the weight of each output    
    lossWeights = {"DWM_output": 1.0, "PVWM_output": 1.0, "GCA_output": 1.0}

    # set up the model

    print("[INFO] compiling model...")
    adam = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    sgd = SGD(lr=0.01, decay=1e-07, momentum=0.7, nesterov=True)

    nn_model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                     metrics=["categorical_accuracy"])
    nn_model.summary()

    # earlystopping use
    checkpoint_name = '%s{epoch:03d}-{loss:.5f}-{val_loss:.5f}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5' % weight_dir
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, mode='auto', save_best_only=True)
    tensor_log = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)

    callbacks_list = [checkpoint, tensor_log, reduce_lr]

    # train parameters: set to appropriate values
    batch_sz = 32
    num_epochs = 100
    steps = int(y_train.shape[0] / batch_sz)
    val_steps = 1
    print('steps per epoch: ', steps, val_steps)

    # data generators
    train_generator = DataGenerator(root, list_IDs_train, y_train, batch_size=batch_sz)
    val_generator = DataGenerator(root, list_IDs_val, y_val, batch_size=batch_sz)

    # fit and save the model
    nn_model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=num_epochs,
                           callbacks=callbacks_list, verbose=1)

    save_model(nn_model, os.path.join(models_dir, '%s.hdf5' % model_name))


# from typing import List


def unet(inputs: Input, pretrained_weights=None, k_reg=None):
    # 1
    conv1 = Conv2D(64, 3, name='conv1', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(inputs)
    conv1 = Conv2D(64, 3, name='conv2', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(conv1)
    pool1 = MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

    # 2
    conv2 = Conv2D(128, 3, name='conv3', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(pool1)
    conv2 = Conv2D(128, 3, name='conv4', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(conv2)
    pool2 = MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

    # 3
    conv3 = Conv2D(256, 3, name='conv5', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(pool2)
    conv3 = Conv2D(256, 3, name='conv6', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(conv3)
    pool3 = MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

    # 4
    conv4 = Conv2D(512, 3, name='conv7', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(pool3)
    conv4 = Conv2D(512, 3, name='conv8', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(conv4)
    drop4 = Dropout(0.5, name='drop4')(conv4)
    pool4 = MaxPooling2D(name='pool4', pool_size=(2, 2))(drop4)

    # 5
    conv5 = Conv2D(1024, 3, name='conv9', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(pool4)
    conv5 = Conv2D(1024, 3, name='conv10', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(conv5)
    drop5 = Dropout(0.5, name='drop5')(conv5)

    # features
    conv6 = Conv2D(1, 1, name='conv11', activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=k_reg)(drop5)

    flat = Flatten(name='flatten')(conv6)
    return flat


def build_model(net_name: str, input_shape: (int, int, int)):
    input_layer = Input(input_shape, name='Input')

    if net_name == 'unet':
        features = unet(inputs=input_layer, pretrained_weights=None, k_reg=None)
    else:
        raise ValueError("name '%s' is not defined " % net_name)

    dwm = get_classifier(features, num_class=4, dense_layer_name='DWM_dense', output_name='DWM_output')
    pvwm = get_classifier(features, num_class=4, dense_layer_name='PVWM_dense', output_name='PVWM_output')
    gca = get_classifier(features, num_class=4, dense_layer_name='GCA_dense', output_name='GCA_output')

    model = Model(inputs=input_layer,
                  outputs=[dwm, pvwm, gca],
                  name=net_name)
    return model


def get_classifier(features, num_class: int, dense_layer_name: str, output_name: str, k_reg=None):
    d1 = Dense(num_class, kernel_regularizer=k_reg, name=dense_layer_name)(features)
    output = Activation('softmax', name=output_name)(d1)
    return output


# root = 'F:/Kanaz/Dementia/'
# data_path = os.path.join(root, 'Data/AIMIN_Challenge_Training_Dataset/Selected_Data')
# output_path = os.path.join(root, 'results')
# os.makedirs(output_path, exist_ok=True)

# model_params = ['DemNet']

# train_model(root, data_path, model_params)

def get_labels(csv_path):
    csv_file_info = pd.read_csv(csv_path)
    dwm = csv_file_info['DWM'].tolist()
    pvwm = csv_file_info['PVWM'].tolist()
    gca = csv_file_info['GCA'].tolist()
    labels = []
    for i in range(len(dwm)):
        labels.append([dwm[i], pvwm[i], gca[i]])
    return labels


if __name__ == '__main__':
    # model = build_model('unet', (256, 256, 24))
    root = './'
    train_model(root, './', ['unet'])
