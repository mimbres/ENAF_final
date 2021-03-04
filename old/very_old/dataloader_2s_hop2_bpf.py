#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataloader.py: 

USAGE:
    import glob
    fps = glob.glob('../fingerprint_dataset/**/**/*.wav') 
    ds = build_dataset(fps)
    it = iter(ds)
    data = it.get_next()
    
def build_dataset(...):
    - For trining embeddings, use build_dataset(...).
    - shuffle and batch are applied with initialization.

def test_build_dataset_and_fname_frame_index(...):
    - Use for testing or buidling DB.
    - Returns unshuffled consecutive STFT segments of 1s with 50% overlap.  
    - shuffle and batch are not applied.
    
@author: skchang@cochlear.ai, Last update: 21 Jun 2019 
"""
import tensorflow as tf
from utils.audio_features_nnfp import AudioFeat 
from utils.math_util import closest_pow2

# Hardware Param.
N_CPU = tf.data.experimental.AUTOTUNE # 4
SHUFFLE_BUFFER_SZ = 1000 #tf.data.experimental.AUTOTUNE # 1000

# STFT Param.
SR = 8000 # {8000, 16000, 32768}
FMIN = 1500
FMAX = 3500
NFFT = 512 # 64 ms
HOP = 256 # 32 ms
PWR = 2.
WIN = 'hann'
AMP_DB = True  

# Slice & Segement Param. (Segs in Slice)
N_POS_SEG_FROM_SLICE = 5 # almost fixed..
SEG_DUR_IN_SEC =  2.# 1(default) or 2 s
SLICE_HOP_IN_SEC = 1#2 * 1.22 #0.5



def build_dataset(fps=list(),
                  batch_sz=int(),
                  shuffle=True,
                  feat='spec'):
    """
    Arguements
    ----------
    - fps: (list) file paths
    - bathch_sz: (int)
    - shuffle: (bool)
    - feat: 'melspec' or 'spec'
    
    Returns
    -------
    - dataset: <tf.data.Dataset>
    
    """
    
    """ We will extract multiple overlapped-segments(SEG_WIN) for a single slice(SLICE_WIN)..."""  
    SLICE_WIN = round(SR * (SEG_DUR_IN_SEC * 1.22) / HOP) * HOP # 1.2 s: we will later crop 1s, 44 stft frames
    SLICE_HOP = SLICE_WIN#closest_pow2(round(SR * SLICE_HOP_IN_SEC)) # 0.512 s; 4096
    SLICE_STFT_FRAMES = round(SLICE_WIN / HOP) # 38
    SEG_WIN = closest_pow2(round(SR * SEG_DUR_IN_SEC / HOP )) # 1.024 s: 32 stft frames
    SEG_HOP = round((SLICE_STFT_FRAMES - SEG_WIN) / (N_POS_SEG_FROM_SLICE - 1)) # 12.5 ms: 
    #SLICE_OFFSET = tf.random.uniform([], minval=0, maxval=SLICE_HOP-1, dtype=tf.int32) 
    SLICE_OFFSET = 0

    
    # Construct audio feature extractor object
    Af = AudioFeat()
    
    
    # Get list of filepaths
    dataset = tf.data.Dataset.from_tensor_slices(fps) # 
    if shuffle: dataset = dataset.shuffle(SHUFFLE_BUFFER_SZ) # (fp)
    
    # Load audio by each
    dataset = dataset.map(Af.load_audio) # (x, sr)
    
    
    # [Get slices] --> [STFT] --> [normalize]
    def _get_slices(x, slice_win, hop_sz, offset):
        """
        FUTURE WORK: Requires to check total sample length here... For now, 
                     audio smaples shorter than 1.x seconds will cause error!! 
        """
        x = tf.signal.frame(x[:, offset:], slice_win, hop_sz) # (1,N,L): N slices, L length        
        x = tf.squeeze(x) # (N,L)
        return tf.data.Dataset.from_tensor_slices(x)
    
    dataset = dataset.flat_map(lambda _x, _sr: _get_slices(_x, SLICE_WIN, SLICE_HOP,
                                                           SLICE_OFFSET)) # x_slice: (L,)
    dataset = dataset.map(lambda _x_sl: tf.reshape(_x_sl, [1, -1])) # (1, L)

    if feat=='spec':
        dataset = dataset.map(lambda _x_sl: Af.spectrogram(_x_sl, NFFT, HOP, PWR, WIN,
                                                           frange=[FMIN, FMAX, SR],
                                                           return_db_gram=AMP_DB,
                                                           reduce_dc=True),
                              num_parallel_calls=N_CPU) # (256, 38, 1)
    elif feat=='melspec':
        dataset = dataset.map(lambda _x_sl: Af.melspectrogram(_x_sl, 1024, HOP, SR,
                                                              nmels=256, fmin=FMIN, fmax=FMAX,
                                                              return_db_melgram=AMP_DB,
                                                              reduce_dc=True),
                              num_parallel_calls=N_CPU) # (256, 38, 1)
 
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
            '/device:GPU:{}'.format(0)))
    
    def _normalize(x):
        x = x - tf.math.reduce_min(x)
        return tf.math.divide_no_nan(x, tf.math.reduce_max(x))
    
    dataset = dataset.map(_normalize) # x_sl: (256, 38, 1) 
    
    
    # Shuffle before positive-sampling...
    if shuffle: dataset = dataset.shuffle(SHUFFLE_BUFFER_SZ) # x_sl: (256, 38, 1) 
    
    
    # Sample N positive segments from a slice
    def _sample_n_segments(x, seg_win, seg_hop):
        x = tf.squeeze(x) # (256, 38)        
        x = tf.signal.frame(x, seg_win, seg_hop, axis=1) # (256, N_slices=7, 32)
        return tf.transpose(x, perm=[0,2,1]) # (256, 32, N_slices=7)
    
    dataset = dataset.map(lambda _x_sl: _sample_n_segments(_x_sl, SEG_WIN, SEG_HOP)) # (256, 32, N_slices=7)
    
    
    # Get batch with 'B = batch_sz = _b * N_slices'
    _b = round(batch_sz / N_POS_SEG_FROM_SLICE) 
    if _b <= 0: raise ValueError('batch_sz must be larger than {}'
                                 .format(N_POS_SEG_FROM_SLICE))
    dataset = dataset.batch(_b, drop_remainder=True) # (_b, 256, 32, N_slices=7)
    
    # Unpack N_slices as batch, so as to 'B = batch_sz * N_slices'
    def _unpack_n_slices_as_batch(x):
        x = tf.transpose(x, [0,3,1,2]) # (b,Nsl,256,32)
        x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3]]) # (B,256,32)
        return tf.expand_dims(x, axis=3) # (B,256,32,1)
    
    dataset = dataset.map(_unpack_n_slices_as_batch) #(B,256,32,1)
    
    
    # Generate temporary labels
    def _add_labels(x): 
        _b = tf.cast(tf.shape(x)[0] / N_POS_SEG_FROM_SLICE, tf.int32) 
        labels = tf.keras.backend.repeat_elements(tf.range(_b), N_POS_SEG_FROM_SLICE, axis=0)
        return (x, labels) # Output tuple
    dataset = dataset.map(_add_labels)
    
    
    # Prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #batch_sz * 4)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
            '/device:GPU:{}'.format(0)))
    
    return dataset # Tuple: ([B,F,T,1], [B])   






def build_dataset_and_fname_frame_index(fps=list(), feat='spec', prefetch=True):
    """    
    Arguements
    ----------
    - fps: (list) file paths
    - feat: 'melspec' or 'spec'
    
    Returns
    -------
    - dataset: <tf.data.Dataset>
    """
    SEG_WIN = closest_pow2(round(SR * SEG_DUR_IN_SEC / HOP )) # 1.024 s: 32 stft frames
    SEG_HOP = closest_pow2(round(SR * SLICE_HOP_IN_SEC / HOP )) # 0.512 s: 16 stft frames
    
    
    # Construct audio feature extractor object
    Af = AudioFeat()
        
    # Get list of filepaths
    dataset = tf.data.Dataset.from_tensor_slices(fps) # 
    
    # Load audio by each
    dataset = dataset.map(lambda _fname: (Af.load_audio(_fname)[0], _fname)) # (x, fname)
    

    # [STFT] --> [Slice]
    if feat=='spec':
        dataset = dataset.map(lambda _x, _fname: (Af.spectrogram(_x, NFFT, HOP, PWR, WIN,
                                                        frange=[FMIN, FMAX, SR],
                                                        return_db_gram=AMP_DB,
                                                        reduce_dc=True), _fname),
                              num_parallel_calls=N_CPU) # (256, n_stft_frames, 1)
    elif feat=='melspec':
        dataset = dataset.map(lambda _x, _fname: (Af.melspectrogram(_x, 1024, HOP, SR,
                                                           nmels=256, fmin=FMIN, fmax=FMAX,
                                                           return_db_melgram=AMP_DB,
                                                           reduce_dc=True), _fname),
                              num_parallel_calls=N_CPU) # (256, n_stft_frames, 1)
    
    
    # Prefetch
    if prefetch:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #batch_sz * 4)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
                '/device:GPU:{}'.format(0)))      
    
    
    def _get_slices(x, slice_win, hop_sz):
        """
        FUTURE WORK: Requires to check total sample length here... For now, 
                     audio smaples shorter than 1.x seconds will cause error!! 
        """
        x = tf.squeeze(x) # (256, n_stft_frames)
        x = tf.signal.frame(x, slice_win, hop_sz) # (256, n_seg, n_frames_in_seg=SEG_WIN)       
        x = tf.transpose(x, perm=[1,0,2]) # (n_seg, 256, n_frames) = (N,F,T)
        return tf.expand_dims(x, axis=3) # (N, F, T, 1)
        
    dataset = dataset.map(lambda _x, _fname: (_get_slices(_x, SEG_WIN, SEG_HOP),
                                              _fname)) # ((N,F,T,1), fname) = ((57, 256, 32, 1), [])
     
    
    # Generate fname_list and frame_list, and slice into frame-wise dataset
    def _gen_fname_frame_list(x, fname): # x:(N,F,T,1), fname:scalar
        x_sz = tf.shape(x)[0]
        frame_list = tf.range(0, x_sz)
        fname_list = tf.fill([x_sz], fname)
        return tf.data.Dataset.from_tensor_slices((x, fname_list, frame_list)) # ((F,T,1), [], [])

    dataset = dataset.flat_map(lambda _x, _fname: _gen_fname_frame_list(_x, _fname)) # ((F,T,1), (), ())


    # Normalize
    def _normalize(x):
        x = x - tf.math.reduce_min(x)
        return tf.math.divide_no_nan(x, tf.math.reduce_max(x))
    
    dataset = dataset.map(lambda _x, _fname, _frame: (_normalize(_x), _fname, _frame))   
    
    # Prefetch
    if prefetch:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #batch_sz * 4)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
                '/device:GPU:{}'.format(0))) 
    return dataset # ((F,T,1), fname, frame)






@tf.function
def test_build_dataset(display_all=False, feat='melspec'):
    fps = ['./utils/small_dataset/001040.mp3_8k.wav',
           './utils/small_dataset/001039.mp3_8k.wav']
    dataset = build_dataset(fps, batch_sz=9, feat=feat) # mel-spec

    if display_all:
        for i, d in enumerate(dataset):
            tf.print(i, d)
    else: # take one example with batchsize=3
        it = iter(dataset)
        d = it.get_next()
    return d # Tuple: ([14,256,32,1], [14])


@tf.function
def test_build_dataset_and_fname_frame_index(display_all=False, feat='spec'):
    fps = ['./utils/small_dataset/001040.mp3_8k.wav',
           './utils/small_dataset/001039.mp3_8k.wav']
    dataset = build_dataset_and_fname_frame_index(fps, feat=feat) # mel-spec

    if display_all:
        for i, d in enumerate(dataset):
            tf.print(i, d)
    else: # take one example with batchsize=3
        it = iter(dataset)
        d = it.get_next()
    return d # ((256, 32, 1), (fname), (frame_idx)) 


#%timeit -n 20 test_build_dataset(False, 'spec') # {spec(49.3), mel(84.9)}ms with @tf.function
