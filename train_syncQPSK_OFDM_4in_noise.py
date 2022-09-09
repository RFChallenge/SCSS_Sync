import os
os.environ['PYTHONHASHSEED'] = '0'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random 
import tensorflow as tf
from tqdm import tqdm
from scipy import signal as sg

import rfcutils # this corresponds to utility functions provided for the challenge

import rfcutils.ofdm_helper_fn_short as ofdmfn

from src import unet_model as unet
from src.time_proc import long_window

get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)

all_sig_mixture, all_sig1, all_sig2, all_sync_idx2, all_sig1_syms, all_sig1_bits, all_start_idx2 = pickle.load(open('dataset/Ex3C3_QPSK_OFDMQAM16.pickle','rb'))
all_val_sig_mixture, all_val_sig1, all_val_sig2, all_val_sync_idx2, all_val_sig1_syms, all_val_sig1_bits, all_val_start_idx2 = pickle.load(open('dataset/Ex3C3_ValSet_QPSK_OFDMQAM16.pickle','rb'))

sig_len_qpsk = 40960 # This is equal to 40960*6/8. Needed for the decimation
all_sinr = np.arange(-24, 4, 1.5)
random.seed(3)
np.random.seed(3)
tf.random.set_seed(3)
n_per_sinr_tr = 1000
n_per_sinr_val = 100
training_examples = 900
val_examples = 100
target_snr = 10
window_len = 40960
seg_len = 0

all_sig_mixture, all_sig1, all_sig2 = [], [] , []
for j in tqdm(np.arange(len(all_sinr))):
    for i in range(training_examples):
        target_sinr = all_sinr[i]
        start_idx2 = np.array(all_sync_idx2[j*n_per_sinr_tr + i])
        all_sig1_ex = np.array(all_sig1[j*n_per_sinr_tr + i])
        sig1 = all_sig1_ex[:window_len]
        all_sig2_ex = np.array(all_sig2[j*n_per_sinr_tr + i])
        sig2 = all_sig2_ex[:window_len]
        tau_b = (80 - start_idx2%80)%80
        
        CNnoise = np.empty(sig1.shape, dtype=np.complex128)
        CNnoise.real = np.random.normal(size=sig1.shape)/np.sqrt(2)
        CNnoise.imag = np.random.normal(size=sig1.shape)/np.sqrt(2)
        
        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        coeff_noise = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(CNnoise)**2)*(10**(target_snr/10))))

        sig_mixture = sig1 + sig2 * coeff + CNnoise * coeff_noise        
        
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig1)

for j in tqdm(np.arange(len(all_sinr))):
    for i in range(training_examples):
        target_sinr = all_sinr[i]
        start_idx2 = np.array(all_val_sync_idx2[j*n_per_sinr_val + i])
        all_sig1_ex = np.array(all_val_sig1[j*n_per_sinr_val + i])
        sig1 = all_sig1_ex[:window_len]
        all_sig2_ex = np.array(all_val_sig2[j*n_per_sinr_val + i])
        sig2 = all_sig2_ex[:window_len]
        tau_b = (80 - start_idx2%80)%80
        
        CNnoise = np.empty(sig1.shape, dtype=np.complex128)
        CNnoise.real = np.random.normal(size=sig1.shape)/np.sqrt(2)
        CNnoise.imag = np.random.normal(size=sig1.shape)/np.sqrt(2)
        
        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        coeff_noise = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(CNnoise)**2)*(10**(target_snr/10))))

        sig_mixture = sig1 + sig2 * coeff + CNnoise * coeff_noise
        
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig1)
        
all_sig_mixture = np.array(all_sig_mixture)
all_sig1 = np.array(all_sig1)


sig1_out = all_sig1.reshape(-1,window_len)
out1_comp = np.dstack((sig1_out.real, sig1_out.imag))

all_mixture_seg = long_window(all_sig_mixture, window_len, seg_len)
mixture_bands_comp = np.dstack((all_mixture_seg.real, all_mixture_seg.imag))

mixture_input_nn = np.dstack((mixture_bands_comp, mixture_bands_comp))

print(f'Output shape: {out1_comp.shape}; Input shape: {mixture_input_nn.shape}')

long_k_sz = 101
model_name = f'ofdm_{window_len}_K{long_k_sz}_XL_TS1000_sync_4in_noisy_{target_snr}_m0'
print(f'Training {model_name}')
nn_model = unet.get_unet_model_XL_4((window_len+seg_len, 4), k_sz=3, long_k_sz=long_k_sz, start_idx=seg_len//2, window_len=window_len)

checkpoint_filepath = f'./tmp_checkpoints/ofdm_{window_len}_K{long_k_sz}_XL_TS1000_sync_4in_noisy_{target_snr}_m0'
IR_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

nn_model.fit(mixture_input_nn, out1_comp, epochs=2000, batch_size=32, validation_split=0.1,shuffle=True, verbose=1,callbacks=[stop_early,IR_model_checkpoint_callback])
