import sys, os
os.environ['PYTHONHASHSEED'] = '0'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random 
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from scipy import signal as sg
# from sklearn.utils import shuffle

import rfcutils # this corresponds to utility functions provided for the challenge

from src import unet_model as unet
from src.time_proc import long_window

get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)

all_sig_mixture, all_sig1, all_sig2, all_sync_idx2, all_sig1_syms, all_sig1_bits, all_start_idx2 = pickle.load(open('dataset/Ex3C3_QPSK_OFDMQAM16.pickle','rb'))
all_val_sig_mixture, all_val_sig1, all_val_sig2, all_val_sync_idx2, all_val_sig1_syms, all_val_sig1_bits, all_val_start_idx2 = pickle.load(open('dataset/Ex3C3_ValSet_QPSK_OFDMQAM16.pickle','rb'))

# Grab the arguments that are passed in
shift = int(sys.argv[1])
print(f'Shift = {shift}')

# Other parameters
all_sinr = np.arange(-30, 4, 1.5)
n_per_sinr_tr = 1000
n_per_sinr_val = 100
training_examples = 400
val_examples = 100
seq_len = 10240
window_len = 10240
eff_train_ex = int(training_examples*seq_len/window_len)

all_sig1_tr_val = []
all_mixture_tr_val = []

for j in tqdm(np.arange(len(all_sinr))):
    for i in range(training_examples):
    
        start_idx2 = np.array(all_sync_idx2[j*n_per_sinr_tr + i])
        all_sig1_ex = np.array(all_sig1[j*n_per_sinr_tr + i])
        all_sig1_ex_w = all_sig1_ex[start_idx2+shift:seq_len+start_idx2+shift]
        all_sig2_ex = np.array(all_sig2[j*n_per_sinr_tr + i])
        all_sig2_ex_w = all_sig2_ex[start_idx2+shift:seq_len+start_idx2+shift]
        all_sig_mixture_ex = np.array(all_sig_mixture[j*n_per_sinr_tr + i])
        all_sig_mixture_ex_w = all_sig_mixture_ex[start_idx2+shift:seq_len+start_idx2+shift]
        
        
        all_sig1_tr_val.append(all_sig1_ex_w)
        all_mixture_tr_val.append(all_sig_mixture_ex_w)
  
for j in tqdm(np.arange(len(all_sinr))):
    for i in range(val_examples):
        
        start_idx2 = np.array(all_val_sync_idx2[j*n_per_sinr_val + i])
        all_sig1_ex = np.array(all_val_sig1[j*n_per_sinr_val + i])
        all_sig1_ex_w = all_sig1_ex[start_idx2+shift:seq_len+start_idx2+shift]
        all_sig2_ex = np.array(all_val_sig2[j*n_per_sinr_val + i])
        all_sig2_ex_w = all_sig2_ex[start_idx2+shift:seq_len+start_idx2+shift]
        all_sig_mixture_ex = np.array(all_val_sig_mixture[j*n_per_sinr_val + i])
        all_sig_mixture_ex_w = all_sig_mixture_ex[start_idx2+shift:seq_len+start_idx2+shift]
        
        all_sig1_tr_val.append(all_sig1_ex_w)
        all_mixture_tr_val.append(all_sig_mixture_ex_w)

all_sig1_tr_val = np.array(all_sig1_tr_val)
all_mixture_tr_val = np.array(all_mixture_tr_val)

# window_len = all_mixture_tr_val.shape[1]

sig1_out = all_sig1_tr_val.reshape(-1,window_len)

out1_comp = np.dstack((sig1_out.real, sig1_out.imag))

sig_mixture_out = all_mixture_tr_val.reshape(-1,window_len)
mixture_bands_comp = np.dstack((sig_mixture_out.real, sig_mixture_out.imag))

print(f'Output shape: {out1_comp.shape}; Input shape: {mixture_bands_comp.shape}')

long_k_sz = 101
model_name = f'qpsk_ofdm64_W{window_len}_TS{eff_train_ex}_K{long_k_sz}_S{shift}'
print(f'Training {model_name}')
nn_model = unet.get_unet_model_XL_2((window_len, 2), k_sz=3, long_k_sz=long_k_sz, start_idx=0, window_len=window_len)
checkpoint_filepath = f'./tmp_checkpoints/checkpoint_qpsk_ofdm64_W{window_len}_TS{eff_train_ex}_K{long_k_sz}_S{shift}'
IR_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, validation_split=0.2,shuffle=True, verbose=1,callbacks=[stop_early,IR_model_checkpoint_callback])