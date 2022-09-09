import warnings
import numpy as np


def eval_ber(bit_est, bit_true):
    if len(bit_est) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_est)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_est))
        bit_true = bit_true[:msg_len]
        bit_est = bit_est[:msg_len]
    ber = np.sum(np.abs(bit_est-bit_true))/len(bit_true)
    return ber

def eval_logloss(bit_prob, bit_true):
    if len(bit_prob) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_prob)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_prob))
        bit_true = bit_true[:msg_len]
        bit_prob = bit_prob[:msg_len]
    logloss = -np.mean((bit_true==0)*np.log2(bit_prob) + (bit_true==1)*np.log2(1-bit_prob))
    return logloss


get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s1, s2: 10*np.log10(get_pow(s1)/get_pow(s2))
get_crosscor = lambda x, y: np.abs(np.correlate(x, y)/np.sqrt(np.correlate(x, x)*np.correlate(y, y)))[0]

from .qpsk_helper_fn import matched_filter_demod
def eval_qpsk_sig(sig1_est, sig1_true, buffer=0):
    assert len(sig1_est) == len(sig1_true), 'Inputs are not of the same length'
    assert buffer == 0 or buffer*2 < len(sig1_true), 'Invalid Buffer length'
    if buffer == 0:
        mse = get_pow(sig1_est - sig1_true)
        sdr = get_sinr(sig1_true, sig1_est - sig1_true)
        cor = get_crosscor(sig1_est, sig1_true)
    else:
        mse = get_pow(sig1_est[buffer:-buffer] - sig1_true[buffer:-buffer])
        sdr = get_sinr(sig1_true[buffer:-buffer], sig1_est[buffer:-buffer] - sig1_true[buffer:-buffer])
        cor = get_crosscor(sig1_est[buffer:-buffer], sig1_true[buffer:-buffer])
    
    bit_est = matched_filter_demod(sig1_est)
    bit_true = matched_filter_demod(sig1_true)
    if len(bit_est) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_est)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_est))
        bit_true = bit_true[:msg_len]
        bit_est = bit_est[:msg_len]
    
    ber = np.sum(np.abs(bit_est-bit_true))/len(bit_true)
    return ber, mse, sdr, cor
    