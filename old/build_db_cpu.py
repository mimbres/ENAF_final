#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:34:50 2019

Speed for fingerprinting 2048 frames(=17min = 3.4songs):
BSZ    TIME(c=2)
32
64     81.35
128    62.78
256    65.40
512    70.50
1024  102.37
2048  207.96

(GPU=1) Speed for fingerprinting 1024*2 embeddings:
BSZ    TIME(c=2)  TIME(c=4)    TIME(c=6)   TIME(c=8)
32      6.03         7.95         6.14       6.18
64     10.23         7.90         6.5        6.02
@author: skchang@cochlear.ai
"""
import sys, glob, os, time
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0"
from EATS import generator_fp
from EATS.networks.kapre2keras.melspectrogram import Melspectrogram
from model.nnfp_l2norm_v2_fixed import FingerPrinter

if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]


# Model, Epoch, N_queries
MODE = 'db'
EXP_NAME = 'exp_v2fix_semihard_320win1_d64_tr100kfull'
EPOCH = '5'
DB_SEL = '100kfull'
SAVED_WEIGHT_DIR = 'logs/checkpoint/' + EXP_NAME
OUTPUT_EMB_DIR = 'logs/emb/' + EXP_NAME + '_' + DB_SEL + '/' + str(EPOCH)
BSZ = 64
N_QUERY = 2000000
FILE_EXT = 'wav'

# Hyper-parameters
FEAT = 'melspec'  # 'spec' or 'melspec'
FS = 8000
DUR = 1
HOP = .5
EMB_SZ = 64  #256 not-working now..

# Directories
if DB_SEL=='100k':
    DATA_ROOT_DIR = '../fingerprint_dataset/music_100k/fma_large_8k/'
elif DB_SEL=='100kfull':
    DATA_ROOT_DIR = '../fingerprint_dataset/music_100k_full/fma_full_8k/'
AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'

music_fps = glob.glob(DATA_ROOT_DIR + '**/*.' + FILE_EXT, recursive=True)
aug_ts_fps = glob.glob(AUG_ROOT_DIR + 'ts/**/*.' + FILE_EXT, recursive=True)
ir_fps = glob.glob(IR_ROOT_DIR + '**/*.' + FILE_EXT, recursive=True)

# Build dataset
def build_dataset(MODE):
    if MODE == 'db':
        ds = generator_fp.genUnbalSequence(
            list(reversed(music_fps)), # reversed list for later on evaluation!!
            BSZ,
            BSZ,
            DUR,
            HOP,
            FS,
            shuffle=False,
            random_offset=False,
            bg_mix_parameter=[False],
            ir_mix_parameter=[False])
    elif MODE == 'query':
        ds = generator_fp.genUnbalSequence(
            list(reversed(music_fps)), # reversed list for later on evaluation!!
            BSZ,
            BSZ,
            DUR,
            HOP,
            FS,
            shuffle=False,
            random_offset=False,
            bg_mix_parameter=[False],
            ir_mix_parameter=[False])
    else:
        raise(NotImplementedError(MODE))
    return ds


# Build & Load model 
input_aud = tf.keras.Input(shape=(1, FS * DUR))
mel = Melspectrogram(
    n_dft=1024,
    n_hop=256,
    sr=FS,
    n_mels=256,
    fmin=300,
    fmax=4000,
    return_decibel_melgram=True)(input_aud)
m_pre = tf.keras.Model(inputs=[input_aud], outputs=[mel])
m_pre.trainable = False
m_fp = FingerPrinter(emb_sz=EMB_SZ, fc_unit_dim=[32, 1])
m_fp.trainable = False

@tf.function
def test_step(X):
    X = m_pre(X)
    X = m_fp(X) # shape: (BSZ, EMB_SZ)
    return X



def gen_emb(ds, MODE):
    os.makedirs(OUTPUT_EMB_DIR, exist_ok=True)
    progbar = tf.keras.utils.Progbar(len(ds))
    
    # Create memmap, and calculate output shape, and save shape
    if MODE=='db':
        output_shape = (len(ds) * BSZ, EMB_SZ)
        output_emb = np.memmap(OUTPUT_EMB_DIR + '/db.mm', dtype='float32', mode='w+', shape=output_shape)
        np.save(OUTPUT_EMB_DIR + '/db_shape.npy', output_shape)
    elif MODE=='query':
        """
        Not implemented
        """
        pass
    else:
        raise(NotImplementedError(MODE))

 
    # Collect embeddings for full-size on-disk DB search, using np.memmap 
    for i, (Xa, Xp) in enumerate(ds):
        progbar.update(i)
        
        if MODE=='db':
            emb = test_step(Xa) # shape: (BSZ, EMB_SZ)
            output_emb[i*BSZ:(i+1)*BSZ, :] = emb.numpy()
        elif MODE=='query':
            """
            Not implemented
            """            
#        if (i*BSZ)==(1024*2):
#            break;
            
    tf.print('------Succesfully saved embeddings to {}-----'.format(OUTPUT_EMB_DIR))
    return


def audit(ds):
    import wavio
    Xa, Xp = ds.__getitem__(0)
    for i in range(BSZ):
        wavio.write('Xa_bak{}.wav'.format(i), Xa[i,0,:], 8000, sampwidth=2)
    
    for i in range(32):
        wavio.write('Xp_bak{}.wav'.format(i), Xp[i,0,:], 8000, sampwidth=2)
    return    


def main():
#    start_time = time.time()
    ds = build_dataset(MODE)
    gen_emb(ds, MODE)
#    print("--- %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    main()
