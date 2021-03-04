#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
printer.py

USAGE:
    $python buildDB.py <EXP_NAME> <optional:EPOCH> <optional:DATA_SELECT>

ARGUMENTS:
    DATA_SELECT: either 'train', 'test', 'all', or 'utube'

OUTPUT:
    This code will create three .npy files to 'logs/fp_output/<EXP_NAME>' directory 
    - 'embs.npy': Q-dimensional numpy vector for N songs (N, Q).
    - 'songid_frame_idx.npy': embeddings to song_id mapping table.

1m55s for 9,000 songs, emb:225Mb, idx:36Mb 
Created on Tue Jul  2 17:51:44 2019

Line 36, 42
@author: skchang@cochlear.ai
"""
import os, sys, glob, random
import tensorflow as tf
import numpy as np
from utils.config_gpu_memory_lim import allow_gpu_memory_growth #config_gpu_memory_limit
# GPU config: This is required if target GPU has smaller vmemory
#allow_gpu_memory_growth() # config_gpu_memory_limit(0, 4)  
#from dataloader import build_dataset_and_fname_frame_index
from dataloader_old_old import build_dataset_and_fname_frame_index
#from model.nnfp import FingerPrinter
from model.nnfp_l2norm import FingerPrinter
from tqdm import tqdm




assert sys.version_info >= (3, 5) # Python ≥3.5 required
assert tf.__version__ >= "2.0"
assert len(sys.argv)>=2, 'python printer.py <EXP_NAME> <optional:EPOCH> <optional:DATA_ROOT>'


# Feature info
BATCH_SZ = 500
FEAT = 'melspec' # 'spec' or 'melspec'
TEST_SPLIT = 0.1

# Data info
EXP_NAME = sys.argv[1] # Required
    
if len(sys.argv)>2: 
    if sys.argv[2]=='':
        EPOCH = None
    else:
        EPOCH = sys.argv[2]
else:
    EPOCH = None # If None, load the latest checkpoint

if len(sys.argv)>3:
    DATASET_SELECT = sys.argv[3]
else:
    DATASET_SELECT = 'all'
    
if (DATASET_SELECT=='train') | (DATASET_SELECT=='test') | (DATASET_SELECT=='all'):
    DATA_ROOT = '../fingerprint_dataset/' # Default data root directory
elif DATASET_SELECT=='utube':
    DATA_ROOT = 'utils/utube_reaction_dataset/'
else:
    raise NotImplementedError(DATASET_SELECT)

# Directory for save embeddings
if EPOCH==None:
    SAVE_RESULT_DIR = 'logs/fp_output/' + EXP_NAME + '-ep-final/' + DATASET_SELECT + '/'
else:
    SAVE_RESULT_DIR = 'logs/fp_output/' + EXP_NAME + '-ep-' + str(EPOCH) + '/' + DATASET_SELECT + '/'
os.makedirs(os.path.dirname(SAVE_RESULT_DIR), exist_ok=True)





#%% Prepare dataset
"""
    [Collect segments for each song] --> [Build song title] --> [Merge]
"""
fps = glob.glob(DATA_ROOT + '**/*.wav', recursive=True) 
random.seed(1)
random.shuffle(fps)
"""
    Select train or test set here... 
"""
if DATASET_SELECT=='train':
    fps = fps[:round(len(fps)*(1-TEST_SPLIT))] # Use train-set
elif DATASET_SELECT=='test':
    fps = fps[round(len(fps)*(1-TEST_SPLIT)):] # Use test-set
else:
    pass;

ds = build_dataset_and_fname_frame_index(fps, feat=FEAT, prefetch=True)
ds = ds.batch(BATCH_SZ)



# Build Model & load weights
model = FingerPrinter()
if EPOCH==None: # Latest checkpoint
    checkpoint_fname = tf.train.latest_checkpoint('./logs/checkpoint/' + EXP_NAME)
else:    
    checkpoint_fname = './logs/checkpoint/' + EXP_NAME + '/ckpt-' + str(EPOCH)
"""
    NOTE: Before loading weights, tf.train.checkpoint should model parameter
          names with model=model! We then restore the parameters. In inference
          mode, we dont need to get optimizer's variable here. 
"""
checkpoint = tf.train.Checkpoint(model=model) # optimizer=opt
checkpoint.restore(checkpoint_fname)



# Functions
@tf.function
def predict(x):
    return model(x)



# Predict embeddings from data 
embs, song_fps, frame_ids = [], [], []

for x, song_fp, frame_id in tqdm(ds):
    embs.append(predict(x).numpy())
    song_fps.append(song_fp.numpy())
    frame_ids.append(frame_id.numpy())
    
embs = np.concatenate(embs, axis=0)
song_fps = np.concatenate(song_fps, axis=0)
frame_ids = np.concatenate(frame_ids, axis=0)
songid_frame_idx = np.transpose(np.vstack((song_fps, frame_ids))) # (N_frame, 2): each column has [fname, frame_idx]



# Save results to disk
np.save(SAVE_RESULT_DIR + 'embs.npy', embs.astype(np.float32))
np.save(SAVE_RESULT_DIR + 'songid_frame_idx.npy', songid_frame_idx, allow_pickle=True)



# Test code
def test_code():
    # GPU config: This is required if target GPU has smaller vmemory
    allow_gpu_memory_growth() # config_gpu_memory_limit(0, 4)  
    
    BATCH_SZ = 3
    FEAT = 'melspec' # 'spec' or 'melspec'
    fps = ['./utils/small_dataset/001039.mp3_8k.wav', './utils/small_dataset/001040.mp3_8k.wav']
    
    # Prepare data
    ds = build_dataset_and_fname_frame_index(fps, feat=FEAT, prefetch=True).batch(BATCH_SZ)
    
    # Load model 
    model = FingerPrinter()
    
    # Decorated prediction function
    @tf.function
    def predict(x):
        return model(x)
    
    # Generate embeddings
    ds_iter = iter(ds)
    embs, song_fps, frame_ids = [], [], []
    
    for i in range(5):
        x, song_fp, frame_id = ds_iter.get_next() # x:(3,256,32,1), song_fp:(3,), frame_id:(3,)
        embs.append(predict(x).numpy()) # emb: (3,128)
        song_fps.append(song_fp.numpy())
        frame_ids.append(frame_id.numpy())
        
    embs = np.concatenate(embs, axis=0)
    song_fps = np.concatenate(song_fps, axis=0)
    frame_ids = np.concatenate(frame_ids, axis=0)
    songid_frame_idx = np.transpose(np.vstack((song_fps, frame_ids)))
    return
    


