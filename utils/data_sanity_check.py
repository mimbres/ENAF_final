# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Sat Jul 18 13:57:06 2020
@author: skchang@cochlear.ai
"""

import sys, glob, os, time
import numpy as np
import tensorflow as tf
from datetime import datetime
from EATS import generator_fp
from EATS.networks.kapre2keras.melspectrogram import Melspectrogram
from utils.plotter import save_imshow, get_imshow_image
from utils.config_gpu_memory_lim import allow_gpu_memory_growth, config_gpu_memory_limit
# from model.nnfp_l2norm_v2_fixed import FingerPrinter
from model.nnfp_dm import FingerPrinter
from model.online_NTxent_variant_loss import OnlineNTxentVariantLoss
#from model.online_triplet_v2_fixed import Online_Batch_Triplet_Loss
from utils.eval_metric import eval_mini 
allow_gpu_memory_growth()  # GPU config: This is required if target GPU has smaller vmemory
from model.lamb_optimizer import LAMB
# from tensorflow_addons.optimizers.lamb import LAMB
from model.spec_augments.specaug_chain import SpecAugChainer


# Generating Embedding 
GEN_EMB_DATASEL = '10k' # '10k' '100k_full'

# NTxent-parameters
TAU = 0.05 # 0.1 #1. #0.3
REP_ORG_WEIGHTS = 1.0
SAVE_IMG = False
BN = 'layer_norm2d' #'batch_norm'
#OPTIMIZER = tfa.optimizers.LAMB #LAMB # tf.keras.optimizers.Adam # LAMB
LR = 1e-4#1e-4#5e-5#3e-5  #2e-5
LABEL_SMOOTH = 0
LR_SCHEDULE = ''

# Hyper-parameters
FEAT = 'melspec'  # 'spec' or 'melspec'
FS = 8000
DUR = 1
HOP = .5
EMB_SZ = 128  #256 not-working now..
TR_BATCH_SZ = 1280 # 640#320 #120#240 #320 #80  #40
TR_N_ANCHOR = 640 #320#160 #60 #120 #64 #16  # 8
VAL_BATCH_SZ = 128 #120
VAL_N_ANCHOR = 64 #60
TS_BATCH_SZ = 320 #160 #160
TS_N_ANCHOR = 160 #32 #80
MAX_EPOCH = 100
EPOCH = ''
TEST_MODE = ''

# Directories
DATA_ROOT_DIR = '../fingerprint_dataset/music_100k/'
AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'

music_fps = sorted(glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True))
aug_tr_fps = sorted(glob.glob(AUG_ROOT_DIR + 'tr/**/*.wav', recursive=True))
aug_ts_fps = sorted(glob.glob(AUG_ROOT_DIR + 'ts/**/*.wav', recursive=True))
ir_fps = sorted(glob.glob(IR_ROOT_DIR + '**/*.wav', recursive=True))


#%%
# Dataloader
def get_train_ds():
    ds = generator_fp.genUnbalSequence(
        fns_event_list=music_fps[:8500],
        bsz=TR_BATCH_SZ,
        n_anchor= TR_N_ANCHOR, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor 
        duration=DUR,  # duration in seconds
        hop=HOP,
        fs=FS,
        shuffle=True,
        random_offset=True,
        bg_mix_parameter=[True, aug_tr_fps, (10, 10)],
        ir_mix_parameter=[True, ir_fps])
    return ds


for ep in range(10):
    print("EPOCH: ", ep)
    train_ds = get_train_ds()
    enq = tf.keras.utils.OrderedEnqueuer(train_ds, use_multiprocessing=True, shuffle=train_ds.shuffle)
    enq.start(workers=8, max_queue_size=14)
    i=0
    while i < len(enq.sequence):
        i += 1
        Xa, Xp = next(enq.get())
        # Xa, Xp = tf.constant(Xa.astype(np.float32)), tf.constant(Xp.astype(np.float32))
        # Check nan
        if np.any(np.isnan(Xa)):
            print("FOUND NAN in Xa it= ", i)
        if np.any(np.isnan(Xp)):
            print("FOUND NAN in Xp it= ", i)    
            
    enq.stop()
        
        
