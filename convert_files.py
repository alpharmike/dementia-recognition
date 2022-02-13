import pydicom as dicom
import imageio
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from data_generator import DataGenerator
from keras.preprocessing.image import ImageDataGenerator
from data_management import load_data


# ds = dicom.dcmread('6618.dcm')
#
# cv2.imwrite('6618.png', pixel_array_numpy)

def convert_image(image_path, dwm, pvwm, gca, dwm_pred, pvwm_pred, gca_pred):
    scn = imageio.volread(image_path, 'DICOM')
    print(scn.shape)
    nrows = 2
    ncols = 4
    patient_id = str(image_path.split('/')[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    fig.suptitle(
        'Patient ID: %s\nLabels: DWM %d    PVWM %d     GCA %d\nPredicted Labels: DWM %d    PVWM %d     GCA %d' % (
        patient_id, dwm, pvwm, gca, dwm_pred, pvwm_pred, gca_pred), fontsize=16)
    for i in range(nrows):
        for j in range(ncols):
            idx = int(i * ncols + j)
            if idx < scn.shape[0]:
                img = scn[idx, :, :]
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].set_title('%d' % idx)
    save_path = image_path + 'with' + '.png'
    fig.savefig(save_path)


#
#
# base_path = 'AIMIN Challenge - Training Dataset'

# y_train = load_data('y_train.csv', values=False)


# base_path = 'final_test'
# # test_csv = load_data('y_test.csv', values=False)
# # test_file = pd.read_csv('y_test.csv')
# res_csv = load_data('results_with.csv', values=False)
# res_file = pd.read_csv('results_with.csv')
# res_IDs = res_file['Study ID']
# y_s = np.empty((25, 3), dtype=int)
# # for index, id in enumerate(res_IDs):
# #     y_s[index] = res_IDs.loc[id]
# #     print(test_csv.loc[id][0])
# for file in os.listdir(base_path):
#     patient_file = base_path + '/' + file
#     id = int(file)
#     convert_image(patient_file, res_csv.loc[id][0], res_csv.loc[id][1], res_csv.loc[id][2], res_csv.loc[id][3],
#                   res_csv.loc[id][4], res_csv.loc[id][5])

# for file in os.listdir(base_path):
#     id_path = base_path + '/' + file
#     for new_file in os.listdir(id_path):
#         inner_path = id_path + '/' + new_file
#         for mri_path in os.listdir(inner_path):
#             mri_image_path = inner_path + '/' + mri_path
#             for image in os.listdir(mri_image_path):
#                 img_path = mri_image_path + '/' + image
#                 convert_image(img_path)

# scn = imageio.volread('13579', 'DICOM')
# img = np.load('y_train.csv')

training_info = pd.read_csv('y_train.csv')
training_IDs = training_info['Study ID']
# print(training_IDs)
# image_base_path = 'AIMIN Challenge - Training Dataset'
# DWM_labels = training_info['DWM']
# PVWM_labels = training_info['PVWM']
# GCA_labels = training_info['GCA']

# patient_files = os.listdir(image_base_path)

# for file in patient_files:
#     patient_mri_file_path = image_path + '/' + file
#     patient_mri_file = os.listdir(patient_mri_file_path)
#     patient_mri_file = patient_mri_file[0].split('-')[0]
#     print(patient_mri_file)

# patient_labels_dict = {}
# counter = 0
# for patient_ID in training_IDs:
#     flag = False
#     if str(patient_ID)[:-4] in patient_files:
#         patient_file = os.path.join(image_base_path, str(patient_ID)[:-4])
#         avail_file = os.listdir(patient_file)[0]
#         inner_file = patient_file + '/' + avail_file
#         mri_files_list = os.listdir(inner_file)
#         # os.mkdir('training/' + str(patient_ID))
#         for mri_file in mri_files_list:
#             if 'tra_dark-fluid' in mri_file:
#                 counter += 1
#                 flag = True
#                 break
#             # if flag:
#             #     print(str(patient_ID)[:-4])
#
#                 # mri_images = os.listdir(inner_file + '/' + mri_file)
#                 # for mri_image in mri_images[-8:]:
#                 #     if '.png' not in mri_image:
#                 #         img_path = inner_file + '/' + mri_file + '/' + mri_image
#                 #         shutil.copy(img_path, 'training/' + str(patient_ID))
# print(counter)
# patient_index = training_IDs.index(patient_ID)
# patient_labels_dict[patient_ID] = {'DWM': DWM_labels[patient_index], 'PVWM': PVWM_labels[patient_index], 'GCA': GCA_labels[patient_index]}
# print(patient_labels_dict)

# path = 'test'
#
# for id_path in os.listdir(path):
#     new_path = path + '/' + id_path
#     image_files = os.listdir(new_path)
#     if len(image_files) != 24:
#         print(id_path)

# training_info = pd.read_csv('/content/drive/My Drive/dementia/y_train.csv')
# val_info = pd.read_csv('/content/drive/My Drive/dementia/y_val.csv')
# test_info = pd.read_csv('y_test.csv')
# training_IDs = training_info['Study ID']
# val_IDs = val_info['Study ID']
# test_IDs = test_info['Study ID']
# image_base_path = 'AIMIN Challenge - Training Dataset'
# patient_files = os.listdir(image_base_path)
# for patient_ID in test_IDs:
#     if str(patient_ID)[:-4] in patient_files:
#         patient_file = os.path.join(image_base_path, str(patient_ID)[:-4])
#         avail_file = os.listdir(patient_file)[0]
#         inner_file = patient_file + '/' + avail_file
#         mri_files_list = os.listdir(inner_file)
#         if str(patient_ID) not in os.listdir('test'):
#             os.mkdir('test/' + str(patient_ID))
#         if len(os.listdir('test/' + str(patient_ID))) == 0:
#             for mri_file in mri_files_list:
#                 if 'tra' in mri_file:
#                     mri_images = os.listdir(inner_file + '/' + mri_file)
#                     for mri_image in mri_images[-8:]:
#                         if '.png' not in mri_image:
#                             img_path = inner_file + '/' + mri_file + '/' + mri_image
#                             #                         if str(patient_ID) not in os.listdir('/content/drive/My Drive/dementia/validation'):
#                             shutil.copy(img_path, 'test/' + str(patient_ID))

# y_train = load_data('y_train.csv', values=False)
#
#
# # x = DataGenerator('training', training_IDs, y_train)
# y_s = np.empty((253, 3), dtype=int)
# for index, id in enumerate(training_IDs):
#     y_s[index] = y_train.loc[id]
#     print(y_train.loc[id][0])
#
# print(y_s[:, 0])

# pics = []
# path = 'D:\Dementia\code\AIMIN Challenge - Training Dataset\98050025\980500251101-Brain MRI -C\Series 4-t2_tirm_tra_dark-fluid M'
# for pic in os.listdir(path):
#     img = imageio.imread(path + '/' + pic)
#     print(img.shape)
#     pics.append(img)

# def get_classes(list_IDs, y_file, index):
#     classes = {}
#     for id in list_IDs:
#         classes[str(id)] = y_file.loc[id][index]
#     return classes
#
#
# train_class = get_classes(training_IDs, y_train, 0)
# # print(train_class)
#
#
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0,
#     zoom_range=0)
# train_generator = train_datagen.flow_from_directory(
#     directory="training",
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42,
#     classes=train_class
# )
#
# print(train_generator.class_indices)

# image_base_path = 'AIMIN Challenge - Evaluation Dataset'
# training_info = pd.read_csv('/content/drive/My Drive/dementia/y_train.csv')
# val_info = pd.read_csv('/content/drive/My Drive/dementia/y_val.csv')
# test_info = pd.read_csv('/content/drive/My Drive/dementia/y_test.csv')
# eval_info = pd.read_csv('sample_submission.csv')
# training_IDs = training_info['Study ID']
# val_IDs = val_info['Study ID']
# test_IDs = test_info['Study ID']
# eval_IDs = eval_info['Study ID']
# image_base_path = 'AIMIN Challenge - Onsite Test Dataset'
# new_file = pd.read_csv('sample_submission_onsite.csv')
# new_IDs = new_file['Study ID']
#
# patient_files = os.listdir(image_base_path)
# for patient_ID in new_IDs:
#     if str(patient_ID)[:-4] in patient_files:
#         patient_file = os.path.join(image_base_path, str(patient_ID)[:-4])
#         avail_file = os.listdir(patient_file)[0]
#         inner_file = patient_file + '/' + avail_file
#         mri_files_list = os.listdir(inner_file)
#         if str(patient_ID) not in os.listdir('final_test'):
#             os.mkdir('final_test/' + str(patient_ID))
#         if len(os.listdir('final_test/' + str(patient_ID))) == 0:
#             for mri_file in mri_files_list:
#                 if 'tra_dark-fluid' in mri_file:
#                     mri_images = os.listdir(inner_file + '/' + mri_file)
#                     mri_images = sorted(mri_images)
#                     for mri_image in mri_images[6:14]:
#                         if '.png' not in mri_image:
#                             img_path = inner_file + '/' + mri_file + '/' + mri_image
#                             #                         if str(patient_ID) not in os.listdir('/content/drive/My Drive/dementia/validation'):
#                             shutil.copy(img_path, 'final_test/' + str(patient_ID))




convert_image('D:\dataset\AIMIN Challenge - Training Dataset\98050079\980500791101-Brain MRI -C\Series 5-t2_tirm_tra_dark-fluid M', 0,0,0,0,0,0)
