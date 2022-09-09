import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow import keras
import keras_tuner as kt 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import os

import rfcutils # this corresponds to utility functions provided for the challenge
import math

# get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
# get_pow = lambda s: np.mean(np.abs(s)**2)

from commpy.modulation import PSKModem, QAMModem
from commpy.filters import rrcosfilter, rcosfilter

# Parameters for QPSK
mod_num = 4
mod = PSKModem(mod_num)

rolloff = 0.5
Fs = 25e6
oversample_factor = 16
Ts = oversample_factor/Fs
tVec, sPSF = rrcosfilter(oversample_factor*8, rolloff, Ts, Fs)
tVec, sPSF = tVec[1:], sPSF[1:]
sPSF = sPSF.astype(np.complex64)

# https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

def get_unet_model_XL_2(input_shape, k_sz=3, long_k_sz=101, start_idx=64, window_len=320, lr=0.0003):
    start_neurons = 8
    
    n_window = input_shape[0]
    n_ch = 2

    in0 = layers.Input(shape=input_shape)
    x = in0 

    x = layers.BatchNormalization()(x)
    
    
    conv1 = layers.Conv1D(start_neurons * 16, long_k_sz, activation="relu", padding="same")(x)
    conv1 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling1D(2)(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling1D(2)(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling1D(2)(conv3)
    pool3 = layers.Dropout(0.5)(pool3)
    
    conv4 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling1D(2)(conv4)
    pool4 = layers.Dropout(0.5)(pool4)
    
    conv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool4)
    conv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv5)
    pool5 = layers.MaxPooling1D(2)(conv5)
    pool5 = layers.Dropout(0.5)(pool5)

    # Middle
    convm = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool5)
    convm = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(convm)

    deconv5 = layers.Conv1DTranspose(start_neurons * 16, k_sz, strides=2, padding="same")(convm)
    uconv5 = layers.concatenate([deconv5, conv5])
    uconv5 = layers.Dropout(0.5)(uconv5)
    uconv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(uconv5)
    uconv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(uconv5)

    deconv4 = layers.Conv1DTranspose(start_neurons * 8, k_sz, strides=2, padding="same")(uconv5)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv1D(start_neurons * 8, k_sz, activation="relu", padding="same")(uconv4)
    uconv4 = layers.Conv1D(start_neurons * 8, k_sz, activation="relu", padding="same")(uconv4)
    
    deconv3 = layers.Conv1DTranspose(start_neurons * 4, k_sz, strides=2, padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv1D(start_neurons * 4, k_sz, activation="relu", padding="same")(uconv3)
    uconv3 = layers.Conv1D(start_neurons * 4, k_sz, activation="relu", padding="same")(uconv3)

    deconv2 = layers.Conv1DTranspose(start_neurons * 2, k_sz, strides=2, padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv1D(start_neurons * 2, k_sz, activation="relu", padding="same")(uconv2)
    uconv2 = layers.Conv1D(start_neurons * 2, k_sz, activation="relu", padding="same")(uconv2)

    deconv1 = layers.Conv1DTranspose(start_neurons * 1, k_sz, strides=2, padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv1D(start_neurons * 1, k_sz, activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv1D(start_neurons * 1, k_sz, activation="relu", padding="same")(uconv1)
    
    output_layer = layers.Conv1D(n_ch, 1, padding="same", activation=None)(uconv1)
    
    x_out = output_layer[:,start_idx:start_idx+window_len,:]
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=k.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError())

    return supreg_net

def get_unet_model_XL_4(input_shape, k_sz=3, long_k_sz=101, start_idx=64, window_len=320, lr=0.0003):
    start_neurons = 8
    
    n_window = input_shape[0]
    n_ch = input_shape[1]

    in0 = layers.Input(shape=input_shape)
    x = in0 

    x = layers.BatchNormalization()(x)
    
    
    conv1 = layers.Conv1D(start_neurons * 16, long_k_sz, activation="relu", padding="same")(x)
    conv1 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling1D(2)(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling1D(2)(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling1D(2)(conv3)
    pool3 = layers.Dropout(0.5)(pool3)
    
    conv4 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling1D(2)(conv4)
    pool4 = layers.Dropout(0.5)(pool4)
    
    conv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool4)
    conv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(conv5)
    pool5 = layers.MaxPooling1D(2)(conv5)
    pool5 = layers.Dropout(0.5)(pool5)

    # Middle
    convm = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(pool5)
    convm = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(convm)

    deconv5 = layers.Conv1DTranspose(start_neurons * 16, k_sz, strides=2, padding="same")(convm)
    uconv5 = layers.concatenate([deconv5, conv5])
    uconv5 = layers.Dropout(0.5)(uconv5)
    uconv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(uconv5)
    uconv5 = layers.Conv1D(start_neurons * 16, k_sz, activation="relu", padding="same")(uconv5)

    deconv4 = layers.Conv1DTranspose(start_neurons * 8, k_sz, strides=2, padding="same")(uconv5)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv1D(start_neurons * 8, k_sz, activation="relu", padding="same")(uconv4)
    uconv4 = layers.Conv1D(start_neurons * 8, k_sz, activation="relu", padding="same")(uconv4)
    
    deconv3 = layers.Conv1DTranspose(start_neurons * 4, k_sz, strides=2, padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv1D(start_neurons * 4, k_sz, activation="relu", padding="same")(uconv3)
    uconv3 = layers.Conv1D(start_neurons * 4, k_sz, activation="relu", padding="same")(uconv3)

    deconv2 = layers.Conv1DTranspose(start_neurons * 2, k_sz, strides=2, padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv1D(start_neurons * 2, k_sz, activation="relu", padding="same")(uconv2)
    uconv2 = layers.Conv1D(start_neurons * 2, k_sz, activation="relu", padding="same")(uconv2)

    deconv1 = layers.Conv1DTranspose(start_neurons * 1, k_sz, strides=2, padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv1D(start_neurons * 1, k_sz, activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv1D(start_neurons * 1, k_sz, activation="relu", padding="same")(uconv1)
    
    output_layer = layers.Conv1D(n_ch//2, 1, padding="same", activation=None)(uconv1)
    
    x_out = output_layer[:,start_idx:start_idx+window_len,:]
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=k.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError())

    return supreg_net