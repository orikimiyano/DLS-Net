from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf

smooth = 1.
'''
Self-adjusting module
'''


def area_l(true, seg):
    pos_g = K.flatten(true)
    pos_p = K.flatten(seg)
    mul_p_g = pos_g * pos_p
    area_size = K.sum(pos_g - mul_p_g) + K.sum(pos_p - mul_p_g)

    return area_size


def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.math.log(y_pred)
    return tf.reduce_sum(crossEntropyLoss, -1)


def levelset(true, seg):
    pos_contour = K.flatten(true)
    pos_g = K.flatten(seg)

    con_mask = K.sum(pos_contour)
    con_no_mask = tf.cast(tf.size(pos_contour), dtype=tf.float32) - con_mask

    num_no_mask = K.sum(pos_g - (pos_contour * pos_g))
    num_mask = K.sum(pos_contour * pos_g)

    pix_a_no_mask = num_no_mask / (con_no_mask + smooth)
    pix_a_mask = num_mask / (con_mask + smooth)

    energy_mask = (((tf.abs(1 - pix_a_mask)) ** 2) * num_mask) + (
            ((tf.abs(0 - pix_a_mask)) ** 2) * tf.abs(con_mask - num_mask))
    energy_no_mask = (((tf.abs(1 - pix_a_mask)) ** 2) * num_no_mask) + (
            (tf.abs(0 - pix_a_no_mask) ** 2) * tf.abs(con_no_mask - num_no_mask))
    return energy_mask + energy_no_mask


def CE_plus_levelset(y_true, y_pred):
    alph = 0.7
    return alph * levelset(y_true, y_pred) + (1 - alph) * Cross_entropy_loss(y_true, y_pred)


'''
loss for LUNET
'''


def LUNET_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.

    for i in range(y_pred_n.shape[1]):
        single_loss = CE_plus_levelset(y_true_n[:, i], y_pred_n[:, i])
        total_single_loss += single_loss
    area_v = area_l(y_true_n[:, i], y_pred_n[:, i])
    alph = tf.cast(area_v, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
    alph = -(alph - 1) ** 4 + 10 / 7
    alph = alph * 0.7
    total_single_loss = alph * total_single_loss / 10000
    return total_single_loss


'''
loss for LHD
'''


def LHD_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.
    for i in range(y_pred_n.shape[1]):
        single_loss = Cross_entropy_loss(y_true_n[:, i], y_pred_n[:, i])
        total_single_loss += single_loss

    area_v = area_l(y_true_n[:, i], y_pred_n[:, i])
    alph = tf.cast(area_v, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
    alph = -(alph - 1) ** 4 + 10 / 7
    alph = alph * 0.7
    total_single_loss = (1 - alph) * total_single_loss / 10000

    return total_single_loss
