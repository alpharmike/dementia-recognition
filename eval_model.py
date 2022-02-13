import pandas as pd
import numpy as np
import os
import imageio
import cv2
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_model(model1, model2, data_file: str, image_path: str, save_path: str):
    x, y = load_test_data(data_file, image_path, dim=(256, 256), n_channels=8, n_classes=4)
    pred1 = model1.predict(x)
    pred2 = model2.predict(x)
    # print(pred)
    y_pred = {}
    print(pred1[0])
    yc_pred = []
    # print(pred2[0])
    for c, var in enumerate(y.columns):
        print('c: ' + str(c))
        for pi1, pi2 in zip(pred1[c], pred2[c]):

            v1 = np.amax(pi1)
            i1 = np.where(pi1 == np.amax(pi1))

            v2 = np.amax(pi2)
            i2 = np.where(pi2 == np.amax(pi2))

            if v2 > v1:
                yc_pred.append(i2)
            else:
                yc_pred.append(i1)
        # yc_pred = [max(max(pi1), max(pi2)) for pi1, pi2 in zip(pred1[c], pred2[c])]
        # print(y_pred)
        y_pred['pred_' + var] = yc_pred
        # valc = pd.DataFrame(data=pred[c], columns=range(4), index=y.index)
        # valc.to_csv(os.path.join(save_path, 'pred_%s_values.csv' % var))

    pred_df = pd.DataFrame(y_pred, index=y.index)
    print(pred_df.shape)
    res = pd.concat([y, pred_df], axis=1)

    res.to_csv(os.path.join(save_path, 'results.csv'))
    f1s = []
    for var in y.columns:
        print('---- > %s < ----' % var)
        conf_mat = confusion_matrix(y[var], pred_df['pred_' + var])
        print(conf_mat)

        f1 = f1_score(y[var], pred_df['pred_' + var], average='macro')
        f1s.append(f1)
        print('f score: ', f1)

    print('average f:', np.mean(f1s))


def load_test_data(data_file: str, image_path: str, dim: (int, int), n_channels: int, n_classes: int):
    y_test = pd.read_csv(data_file, index_col=0)
    list_id = y_test.index.tolist()
    num_data = len(list_id)

    X = np.empty((num_data, *dim, n_channels))

    for index, id_ in enumerate(list_id):
        img_file_path = image_path + '/' + str(id_)
        chn = 0
        for img_file in os.listdir(img_file_path):
            if chn < n_channels:
                img_path = img_file_path + '/' + img_file
                image = imageio.imread(img_path)
                image = image.astype('float32')
                res_image = cv2.resize(image, dim)
                cv2.normalize(res_image, None, 0, 255, cv2.NORM_MINMAX)
                X[index, :, :, chn] = res_image
            chn = chn + 1

        # image_file = np.reshape(image_file, )

    return X, y_test


if __name__ == '__main__':
    root = './'
    conv_model_model_file = 'unet-drop_0.3-bs8-7_13.hdf5'
    conv_model_model_path = os.path.join(root, 'NN/Models/', conv_model_model_file)

    dense_model_model_file = 'unet-drop_0.3-bs8-7_13-new.hdf5'
    dense_model_model_path = os.path.join(root, 'NN/Models/', dense_model_model_file)

    # data_path = os.path.join(root, 'Data/AIMIN_Challenge_Training_Dataset/Selected_Data/')
    test_image_path = os.path.join(root, 'test')
    test_file = os.path.join(root, 'y_test.csv')

    prefix = ''
    output_path = os.path.join(root, prefix + 'final_mix_own_test', dense_model_model_file[:-5])
    os.makedirs(output_path, exist_ok=True)

    conv_model = load_model(conv_model_model_path, compile=False)
    conv_model_weight_path = os.path.join(root, 'NN/weights/791-2.4746-4.6411-conv.hdf5')
    conv_model.load_weights(conv_model_weight_path)

    dense_model = load_model(dense_model_model_path, compile=False)
    dense_model_weight_path = os.path.join(root, 'NN/weights/200-2.6144-4.7831-dense.hdf5')
    dense_model.load_weights(dense_model_weight_path)
    evaluate_model(conv_model, dense_model, test_file, test_image_path, output_path)
