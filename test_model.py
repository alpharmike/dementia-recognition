import os

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from Train import build_model
from Utils import weighted_categorical_crossentropy
from data_generator import DataGenerator
from data_management import load_data
import pandas as pd
import numpy as np
from keras.models import save_model, load_model
import imageio
import cv2

weight_path = 'NN/weights/unet/001.hdf5'
model_path = 'NN/Models/unet1.hdf5'
test_info = pd.read_csv('y_test.csv')
test_IDs_list = test_info['Study ID'].tolist()
print(test_IDs_list)
test_file = 'y_test.csv'
# train_file = os.path.join('./', 'y_train.csv')
# y_train = load_data(train_file, values=False)
# y_test = load_data(test_file, values=False)
# print(y_test)
# weights = {}
# resp_vars = ['DWM', 'PVWM', 'GCA']
# for var in resp_vars:
#     y_val = y_train[var].values
#     weights[var] = class_weight.compute_class_weight('balanced', np.unique(y_val), y_val.flatten())
# losses = {
#         "DWM_output": weighted_categorical_crossentropy(np.float32(weights['DWM']), "DWM_loss"),
#
#          "PVWM_output": weighted_categorical_crossentropy(np.float32(weights['PVWM']), "PVWM_loss"),
#          "GCA_output": weighted_categorical_crossentropy(np.float32(weights['GCA']), "GCA_loss")
#      }
# lossWeights = {"DWM_output": 1.0, "PVWM_output": 1.0, "GCA_output": 1.0}
# adam = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
# model = build_model('unet', (256, 256, 24))
# model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
#                      metrics=["categorical_accuracy"])
# model.load_weights(model_path)
# batch_sz = 32
# test_generator = DataGenerator('test2', test_IDs_list, y_test)
# predict = model.evaluate_generator(test_generator, steps=1)
# print(predict)

# pred= model.predict_generator(test_generator)
# print(pred)
# predicted_class_indices=np.argmax(pred)
# labels = (test_generator.class_indices)
# labels2 = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
# print(predicted_class_indices)
# print (labels)
# print (predictions)

# test_datagen = ImageDataGenerator(rescale=1/255.)
#
# test_generator = test_datagen.flow_from_directory('test',
#                               # only read images from `test` directory
#                               classes=['test'],
#                               # don't generate labels
#                               class_mode=None,
#                               # don't shuffle
#                               shuffle=False,
#                               # use same size as in training
#                               target_size=(256, 256))
#
# preds = model.predict_generator(test_generator, steps=1)
# print(preds)
# test_info = pd.read_csv('y_test.csv')
# test_IDs_list = test_info['Study ID'].tolist()
# print(test_IDs_list)
# test_file = 'y_test.csv'
# y_test = load_data(test_file, values=False)
model = load_model(model_path, compile=False)
# model.load_weights(weight_path)


# batch_sz = 32
# test_generator = DataGenerator('/content/drive/My Drive/dementia/test', test_IDs_list, y_test, batch_size=batch_sz)
# pred = model.predict()
# print(pred)
# predicted_class_indices = np.argmax(pred)
# labels = (test_generator.class_indices)
# labels2 = dict((v, k) for k, v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
# print(predicted_class_indices)
# print(labels)
# print(predictions)



def read_data(path, input_size):
    all_test = []
    for patient_file in os.listdir(path):
        patient_path = path + '/' + patient_file
        x_test = np.empty((256, 256, 24))
        for index, image_name in enumerate(os.listdir(patient_path)):
            img_path = patient_path + '/' + image_name
            image = imageio.imread(img_path)
            image = cv2.resize(image, input_size) / 255.0
            x_test[:, :, index] = image
        x_test = np.expand_dims(x_test, axis=0)
        all_test.append(x_test)
    return all_test


x_test_all = read_data('test', (256, 256))
preds = []
for x_test in x_test_all:
    pred = model.predict(x_test)
    preds.append(pred)
for pred in preds:
    print(pred)
