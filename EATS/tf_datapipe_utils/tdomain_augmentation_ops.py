# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Time-domain augmentation operators.

## Examplary flow of time-domain augmentations ##

[Sample random input] --> [Time-offset modulation] --> [BG(env) mix]

--> [BG(speech) mix] --> [Random signal mix I] --> [Room IR (ddsp/recorded_IR)] --> [Mic IR]

--> [Clipping] --> [Random signal mix II]


Created on Wed Sep 16 22:22:04 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf


SR = 8000 # sampling rate
INPUT_LEN_SEC = 1 # in sec.
MAX_SEG_INTERVAL_P = 0.2 # in ratio (0~1.)
SEG_WIN_SEC = INPUT_LEN_SEC * (1 + MAX_SEG_INTERVAL_P) # in sec
SEG_HOP_SEC = INPUT_LEN_SEC / 2. # in sec


"""Helper func."""
def sec_to_sample(s, sr=8000):
    return int(s * sr)


"""Ops for Time-domain augmentations. Part 1"""
# T-domain input sampling with random seg_offset: take 1s from 1.2s
def input_random_sample(x,
                        max_seg_offset_s=MAX_SEG_INTERVAL_P*INPUT_LEN_SEC,
                        n_sampled_output=2,
                        fix_offset_only_first_output=False):
    # fix_offset_only_first_output: this is used for DB test mode. 
    if n_sampled_output == 1:
        if fix_offset_only_first_output:
            sample_start = 0
        else:
            sample_start = tf.random.uniform([], 0, sec_to_sample(max_seg_offset_s),
                                            dtype=tf.dtypes.int32)
        return tf.signal.frame(x[sample_start:], sec_to_sample(INPUT_LEN_SEC),
                            sec_to_sample(INPUT_LEN_SEC), True)[0]
    elif n_sampled_output == 2: # TODO: make this dynamic outside.
        sample_starts = tf.random.uniform([n_sampled_output], 0,
            sec_to_sample(max_seg_offset_s), dtype=tf.dtypes.int32)
        if fix_offset_only_first_output:
            sample_starts[0] = 0
        output1 = tf.signal.frame(x[sample_starts[0]:],
                                  sec_to_sample(INPUT_LEN_SEC),
                                  sec_to_sample(INPUT_LEN_SEC), True)[0]
        output2 = tf.signal.frame(x[sample_starts[1]:],
                                  sec_to_sample(INPUT_LEN_SEC),
                                  sec_to_sample(INPUT_LEN_SEC), True)[0]
        return (output1, output2)
    
    

"""Ops for Time-domain augmentations. Part 2"""
# BG & IR mix ops assume a zipped dataset with{Music, Background & IR}.

# Background mix 
def taug_op_bg_mix(x_bypass, x, xbg, snr_range=[5, 15], rand_master_amp=True):
    """
    x: source audio, music segment (T)
    xbg: background sound segment (T)
    snr_range: SNR in dB
    """
    # Random SNR
    snr_range = tf.cast(snr_range, tf.float32)
    min_snr = tf.minimum(snr_range[0], snr_range[1]) 
    max_snr = tf.maximum(snr_range[0], snr_range[1])
    snr = tf.random.uniform([], min_snr, max_snr)

    x_max = tf.reduce_max(tf.abs(x))
    xbg_max = tf.reduce_max(tf.abs(xbg))
   
    # Mix
    if x_max == 0. or xbg_max == 0.:
        x_mix = x + xbg
    else:
        energy_x = tf.reduce_sum(tf.pow(x/100., 2.))
        x = x / tf.sqrt(energy_x)
        energy_bg = tf.reduce_sum(tf.pow(xbg/100., 2.))
        xbg = xbg / tf.sqrt(energy_bg)

        magnitude = tf.pow(10., snr/20.) # This includes sqrt.
        x_mix = magnitude * x + xbg
        x_mix = x_mix / tf.reduce_max(tf.abs(x_mix))
    
    if rand_master_amp:
        # Random master amp (TODO: replace with log-scale_random_number)
        master_amp = tf.random.uniform([], 0.1, 1.)
        x_mix = x_mix * master_amp

    return (x_bypass, x_mix)

# Impulse response mix 
def taug_op_ir_mix(x_bypass, x, xir,
                   ir_length_range_s=[0.1, 0.5],
                   fb_range=[0.1, 0.9],
                   rand_master_amp=True):
    # ir_dur: IR filter length 
    ir_dur_s = tf.random.uniform([], ir_length_range_s[0], ir_length_range_s[1])  
    ir_dur = sec_to_sample(ir_dur_s)

    nfft = sec_to_sample(INPUT_LEN_SEC)
    x_f = tf.signal.stft(x, nfft, 1, window_fn=None) # output dtype: complex64
    xir_f = tf.signal.stft(xir, nfft, 1, window_fn=None)
    x_feedback = tf.signal.inverse_stft(tf.multiply(xir_f, x_f), nfft, 1,
                                        window_fn=None) # output dtype: float32
    mix_ratio = tf.random.uniform([], fb_range[0], fb_range[1])
    x_mix = x * (1. - mix_ratio) + x_feedback * mix_ratio
    
    if rand_master_amp:
        # Random master amp (TODO: replace with log-scale_random_number)
        master_amp = tf.random.uniform([], 0.1, 1.)
        x_mix = x_mix / tf.reduce_max(tf.abs(x_mix))
        x_mix = x_mix * master_amp

    return (x_bypass, x_mix)

"""Ops for Time-domain augmentations. Part 3"""
# T-domain augmentation: simulation of clipping with batch-wise processing
def taug_clipping(x_bypass, x, prob=0.3, amp_max=10.):
    """
    returns (x):
        x: (T) or (B,T) tensor audio samples 
    """
    if tf.random.uniform([]) < prob:
        # Normalize audio (working with both (T) and (B,T) tensors)
        x = tf.divide(x, tf.reduce_max(tf.abs(x), axis=-1, keepdims=True))
        
        # Amp.
        amp = tf.random.uniform([], minval=1., maxval=amp_max)
        x = x * amp
        x = tf.clip_by_value(x, clip_value_min=-1., clip_value_max=1.)

    return (x_bypass, x) # x: (T) or (B,T)


# T-domain augmentation: simulation of sampling error and aliasing
def taug_downsample_alias(x_bypass, x, prob=0.1, fs_source=8000,
                          fs_target_list=[4000, 6000, 7000],
                          interp='nearest'):
    """
    Args:
        x: (T) tensor audio samples
        prob: (float) activiation probability (0-1.)
        fs_source: (int) sampling rate of input recording
        fs_target_list: downsampling target rate
        interp: (str) ['nearest', 'lanczos5', 'lanczos3', 'bicubic',
                       'mitchellcubic', 'bilinear', 'gaussian']
    returns x:
        x: (T) tensor audio samples
    """  
    if tf.random.uniform([]) < prob:
        # Random fs_target
        fs_sel = tf.random.uniform([], 0, len(fs_target_list), 'int32')
        fs_target = tf.gather(fs_target_list, fs_sel) # fs_target_list[fs_sel]
        source_length = tf.shape(x)[-1]
        downsp_length = (tf.cast(source_length, tf.float32) / fs_source) * tf.cast(fs_target, tf.float32)
        downsp_length = tf.cast(downsp_length, tf.int32)
        # Random aliasing 
        antialias = tf.gather([True, False], tf.random.uniform([], 0, 2, 'int32'))            

        # Apply resampling (downsample --> upsample)
        x = tf.reshape(x, [1, -1, 1])
        if (tf.__version__ >= "2.0"):
            if antialias:
                x = tf.image.resize(x, [1, downsp_length], antialias=True, method=interp)
                x = tf.image.resize(x, [1, source_length], antialias=True, method=interp)
            else:
                x = tf.image.resize(x, [1, downsp_length], antialias=False, method=interp)
                x = tf.image.resize(x, [1, source_length], antialias=False, method=interp)
        else: # TF 1 doesnt' have anti-alias 
            x = tf.image.resize(x, [1, downsp_length], method=interp)
            x = tf.image.resize(x, [1, source_length], method=interp)
        x = tf.reshape(x, [-1])

    return (x_bypass, x) # x: (T) where T is time index


# TEST
