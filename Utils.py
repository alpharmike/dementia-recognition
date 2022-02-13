import keras.backend as K
import numpy as np


def weighted_binary_crossentropy(weights, name):
    w_zero = weights[0]
    w_one = weights[1]

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc
        l = -(y_true * K.log(y_pred) * w_one + (1. - y_true) * K.log(1. - y_pred) * w_zero)

        return l

    loss.__name__ = name
    return loss


def weighted_categorical_crossentropy(weights, name):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss_func(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)

        return loss

    loss_func.__name__ = name
    return loss_func


def MFE(weights, name):
    """ Mean False Error """

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc 
        dy = abs(y_true - y_pred)
        w = y_true * weights[1] + (1 - y_true) * weights[0]
        l = w * dy

        return l

    loss.__name__ = name
    return loss


def MSFE(weights, name):
    """ Mean False Error """

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc 
        dy = abs(y_true - y_pred)
        w = y_true * weights[1] + (1 - y_true) * weights[0]
        l = np.power(w * dy, 2)

        return l

    loss.__name__ = name
    return loss
