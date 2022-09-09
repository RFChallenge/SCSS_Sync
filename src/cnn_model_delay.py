import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import os


def get_cnn_delay_model(input_shape, vec_delays, k_sz=100, lr=0.0001):
    
    in0 = layers.Input(shape=input_shape)
    x = in0
    x = layers.BatchNormalization()(x)
    
    conv1 = layers.Conv1D(vec_delays, k_sz+1, activation="relu", padding="same")(x)
    conv2 = layers.Conv1D(vec_delays//2, k_sz//2+1, activation="relu", strides=2, padding="same")(conv1)
    conv3 = layers.Conv1D(vec_delays//4, k_sz//4+1, activation="relu", strides=2, padding="same")(conv2)

    penultimate_layer = layers.Flatten()(conv3)
    penultimate_layer = layers.Dense(vec_delays,activation='relu')(penultimate_layer)
    output_layer = layers.Dense(vec_delays,activation='softmax')(penultimate_layer)
    
    x_out = output_layer
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=k.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy()])

    return supreg_net

def get_cnn_model_delay_new(input_shape, vec_delays, k_sz=5, long_k_sz=100, lr=0.0001):
    
    in0 = layers.Input(shape=input_shape)
    x = in0
    x = layers.BatchNormalization()(x)
    
    conv1 = layers.Conv1D(vec_delays, long_k_sz, activation="relu", padding="same")(x)
    conv2 = layers.Conv1D(vec_delays//2, long_k_sz//2, activation="relu", strides=2, padding="same")(conv1)
    conv3 = layers.Conv1D(vec_delays//4, long_k_sz//4, activation="relu", strides=2, padding="same")(conv2)

    penultimate_layer = layers.Flatten()(conv3)
    penultimate_layer = layers.Dense(vec_delays,activation='relu')(penultimate_layer)
    output_layer = layers.Dense(vec_delays,activation='softmax')(penultimate_layer)
    
    x_out = output_layer
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=k.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy()])

    return supreg_net