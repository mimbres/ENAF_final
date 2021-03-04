#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:15:30 2019

USAGE:
    CUDA_VISIBLE_DEVICES=0 python train_batch_hard.py <opt1:(str)EXP_NAME> <opt2:(int)EPOCH> <opt3:(str)TEST_MODE>

OPTIONS:
    - If <opt1:EXP_NAME> is given and pretrained model exists:
        - If <opt2> is empty, it continues training from the latest checkpoint.
        - Else if <opt2> is given, it continues training from the <opt2:EPOCH>.
    - If <opt1:EXP_NAME> is given and pretrained model doesn't exist, fresh start training.
    - If <opt1:EXP_NAME> is not given, EXP_NAME will be generated by dd/hh/mm/ss.
    - If <opt1>, <opt2> and <opt3:TEST_MODE> are given:
        - If <opt3:TEST_MODE> == "mini-test-only", it loads pre-trained model and proceed in-memory-search-mini-test, then quit.
        - If <opt3:TEST_MODE> == "mini-test", it tests while training.
        - If <opt3:TEST_MODE> == "save-db", it loads pre-trained model and only saves embeddings for DB as a numpy file.
        - If <opt3:TEST_MODE> == "save-db-query", it loads pre-trained model and saves embeddings for DB and Query as numpy files.
LOG-DIRECTORIES:
    
CUDA_VISIBLE_DEVICES=0 python train_with_EATS_v2fix_batch_cos_semihard_320win1_margin02.py exp_v2fix_cos_semihard_320win1_margin02
CUDA_VISIBLE_DEVICES=0 python train_with_EATS_v2fix_batch_cos_semihard_320win1_margin02.py exp_v2fix_cos_semihard_320win1_margin02 '' mini-test-only
CUDA_VISIBLE_DEVICES=0 python train_with_EATS_v2fix_batch_cos_semihard_320win1_margin02.py exp_v2fix_cos_semihard_320win1_margin02 '' save-db-query # 3:98.51
CUDA_VISIBLE_DEVICES=0 python eval.py logs/emb/exp_v2fix_cos_semihard_320win1_margin02/22 120 argmax


@author: skchang@cochlear.ai
"""
#%%
import sys, glob, os
import numpy as np
import tensorflow as tf
from datetime import datetime
from EATS import generator_fp
from EATS.networks.kapre2keras.melspectrogram import Melspectrogram
from utils.config_gpu_memory_lim import allow_gpu_memory_growth
from model.nnfp_l2norm_v2_fixed import FingerPrinter
#from model.nnfp import FingerPrinter
from model.online_triplet_v2_fixed import Online_Batch_Triplet_Loss
from utils.eval_metric import eval_mini 
if tf.test.is_gpu_available(): allow_gpu_memory_growth() # GPU config: This is required if target GPU has smaller vmemory

# Hyper-parameters
LOSS_MODE = 'semi-hard'#'all-balanced'
FEAT = 'melspec'  # 'spec' or 'melspec'
FS = 8000
DUR = 1
HOP = 0.5
LR = 1e-3  #2e-5
EMB_SZ = 128  #256 not-working now..
TR_BATCH_SZ = 320 #80  #40
TR_N_ANCHOR = 64 #16  # 8
TS_BATCH_SZ = 160
TS_N_ANCHOR = 32
MARGIN = 0.2  #0.5
MAX_EPOCH = 500
EPOCH = ''
TEST_MODE = ''

# Directories
DATA_ROOT_DIR = '../fingerprint_dataset/music/'
AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'

music_fps = glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True)
aug_tr_fps = glob.glob(AUG_ROOT_DIR + 'tr/**/*.wav', recursive=True)
aug_ts_fps = glob.glob(AUG_ROOT_DIR + 'ts/**/*.wav', recursive=True)
ir_fps = glob.glob(IR_ROOT_DIR + '**/*.wav', recursive=True)
#%%
#EXP_NAME = 'exp_v2fix_semihard_240'
#EPOCH = ''
TEST_MODE = 'mini-test'


if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
if len(sys.argv) > 2:
    EPOCH = sys.argv[2]
if len(sys.argv) == 1:
    EXP_NAME = datetime.now().strftime("%Y%m%d-%H%M")
if len(sys.argv) > 3:
    TEST_MODE = sys.argv[3]
#%%
CHECKPOINT_SAVE_DIR = './logs/checkpoint/{}/'.format(EXP_NAME)
CHECKPOINT_N_HOUR = 1  # None: disable
LOG_DIR = "logs/fit/" + EXP_NAME
IMG_DIR = 'logs/images/' + EXP_NAME
EMB_DIR = 'logs/emb/' + EXP_NAME
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Dataloader
train_ds = generator_fp.genUnbalSequence(
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

val_ds = generator_fp.genUnbalSequence(
    music_fps[8500:],
    TS_BATCH_SZ,
    TS_N_ANCHOR,
    DUR,
    HOP,
    FS,
    shuffle=False,
    random_offset=True,
    bg_mix_parameter=[True, aug_ts_fps, (10, 10)],
    ir_mix_parameter=[True, ir_fps])

test_ds = generator_fp.genUnbalSequence(
    list(reversed(music_fps)), # reversed list for later on evaluation!!
    TS_BATCH_SZ,
    TS_N_ANCHOR,
    DUR,
    HOP,
    FS,
    shuffle=False,
    random_offset=True,
    bg_mix_parameter=[True, aug_ts_fps, (10, 10)],
    ir_mix_parameter=[True, ir_fps])

# Define Metric, Summary Log
tr_loss = tf.keras.metrics.Mean(name='train_loss')
ts_loss = tf.keras.metrics.Mean(name='test_loss')
tr_fraction = tf.keras.metrics.Mean(name='train_fraction')
tr_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/train')
ts_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/test')

# Build Model: m_pre, m_fp
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

# Define Optimizer & Loss
opt = tf.keras.optimizers.Adam(learning_rate=LR)
Online_Triplet_Loss_tr = Online_Batch_Triplet_Loss(
    TR_BATCH_SZ,
    TR_N_ANCHOR,
    n_pos_per_anchor=int((TR_BATCH_SZ - TR_N_ANCHOR) / TR_N_ANCHOR),
    use_anc_as_pos=True)
loss_obj_tr = Online_Triplet_Loss_tr.batch_cos_semihard_loss

Online_Triplet_Loss_ts = Online_Batch_Triplet_Loss(
    TS_BATCH_SZ,
    TS_N_ANCHOR,
    n_pos_per_anchor=int((TS_BATCH_SZ - TS_N_ANCHOR) / TS_N_ANCHOR),
    use_anc_as_pos=True)
loss_obj_ts = Online_Triplet_Loss_ts.batch_cos_semihard_loss

# Chekcpoint manager
checkpoint = tf.train.Checkpoint(optimizer=opt, model=m_fp)
c_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_SAVE_DIR, 3,
                                       CHECKPOINT_N_HOUR)

if EPOCH == '':
    if c_manager.latest_checkpoint:
        tf.print("-----------Restoring from {}-----------".format(
            c_manager.latest_checkpoint))
        checkpoint.restore(c_manager.latest_checkpoint)
        EPOCH = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]
    else:
        tf.print("-----------Initializing from scratch-----------")
else:    
    checkpoint_fname = CHECKPOINT_SAVE_DIR + 'ckpt-' + str(EPOCH)
    tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
    checkpoint.restore(checkpoint_fname)
    



# Trainer
@tf.function
def train_step(Xa, Xp):  # {Xa: input_anchor, Xp: input_positive}
    X = tf.concat((Xa, Xp), axis=0)
    feat = m_pre(X)  # (nA+nP, F, T, 1)
    with tf.GradientTape() as t:
        emb = m_fp(feat)  # (B, E)
        loss = loss_obj_tr(
                emb_anchor=emb[:TR_N_ANCHOR, :],
                emb_pos=emb[TR_N_ANCHOR:, :],
                margin=MARGIN,  # margin is not implemented yet
                use_anc_as_pos=True)

    g = t.gradient(loss, m_fp.trainable_variables)
    opt.apply_gradients(zip(g, m_fp.trainable_variables))
    
    # Logging
    tr_loss(loss)
    with tr_summary_writer.as_default():
        tf.summary.scalar('loss', tr_loss.result(), step=opt.iterations)
    return


@tf.function
def val_step(Xa, Xp):
    X = tf.concat((Xa, Xp), axis=0)
    feat = m_pre(X)
    emb = m_fp(feat) 
    loss = loss_obj_ts(
        emb_anchor=emb[:TS_N_ANCHOR, :],
        emb_pos=emb[TS_N_ANCHOR:, :],
        margin=MARGIN,
        use_anc_as_pos=True)
    
    # Logging
    ts_loss(loss)
    with ts_summary_writer.as_default():
        tf.summary.scalar('loss', ts_loss.result(), step=opt.iterations)
    return


@tf.function
def test_step(Xa, Xp):
    X = tf.concat((Xa, Xp), axis=0)
    feat = m_pre(X)
    emb = m_fp(feat)
    #return emb
    return tf.split(emb, [TS_N_ANCHOR, TS_BATCH_SZ - TS_N_ANCHOR], axis=0)  # emb_Anchor, emb_Pos

#%% test functions...
def generate_emb(n_query=100000):
    """
    Arguemnts:
        n_query: Number of embeddings for queries. If 0, then save only DB.
    
    Output Numpy memmap-files:
        logs/emb/<exp_name>/db.mm: (float32) tensor of shape (nFingerPrints, dim)
        logs/emb/<exp_name>/query.mm: (float32) tensor of shape (nFingerprints, nAugments, dim)
        logs/emb/<exp_name>/db_shape.npy: (int) 
        logs/emb/<exp_name>/query_shape.npy: (int)
    
    """
    n_augs_per_anchor = int((TS_BATCH_SZ - TS_N_ANCHOR) / TS_N_ANCHOR)
    n_augs_per_batch = n_augs_per_anchor * TS_N_ANCHOR
    n_query = (n_query // n_augs_per_batch) * n_augs_per_batch 
    n_batch_required_for_queries = int(n_query / n_augs_per_batch)
    
    shape_db = (len(test_ds) * TS_N_ANCHOR, EMB_SZ)
    shape_query = (n_query // n_augs_per_anchor, n_augs_per_anchor, EMB_SZ)
    
    # Create memmap, and save shapes
    os.makedirs(EMB_DIR + '/{}'.format(EPOCH), exist_ok=True)
    db = np.memmap(EMB_DIR + '/{}'.format(EPOCH) + '/db.mm', dtype='float32', mode='w+', shape=shape_db)
    np.save(EMB_DIR + '/{}'.format(EPOCH) + '/db_shape.npy', shape_db)
    if (n_query > 0):
        query = np.memmap(EMB_DIR + '/{}'.format(EPOCH) + '/query.mm', dtype='float32', mode='w+', shape=shape_query)
        np.save(EMB_DIR + '/{}'.format(EPOCH) +'/query_shape.npy', shape_query)
    
    progbar = tf.keras.utils.Progbar(len(test_ds))
    
    # Collect embeddings for full-size on-disk DB search, using np.memmap 
    for i, (Xa, Xp) in enumerate(test_ds):
        progbar.update(i)
        emb_anc, emb_pos = test_step(Xa, Xp)
        db[i*TS_N_ANCHOR:(i+1)*TS_N_ANCHOR, :] = emb_anc.numpy()
        if (i < n_batch_required_for_queries) & (n_query != 0):
            emb_pos = tf.reshape(emb_pos,
                                 (TS_N_ANCHOR, n_augs_per_anchor, EMB_SZ))  # (B,4,128)
            query[i*TS_N_ANCHOR:(i+1)*TS_N_ANCHOR, :, : ] = emb_pos.numpy()
    
    tf.print('------Succesfully saved embeddings to {}-----'.format(EMB_DIR + '/{}'.format(EPOCH)))
    del(db) # close mmap-files
    if (n_query > 0): del(query)
    return


def mini_search_test(n_samples=3000):
    db = np.empty((0, EMB_SZ))
    query = np.empty((0, int((TS_BATCH_SZ - TS_N_ANCHOR) / TS_N_ANCHOR), EMB_SZ))
    n_iter = n_samples // TS_BATCH_SZ
    
    # Collect mini DB
    for i in range(n_iter):
        Xa, Xp = test_ds.__getitem__(i)
        emb_anc, emb_pos = test_step(Xa, Xp)
        emb_pos = tf.reshape(emb_pos, (TS_N_ANCHOR, -1, EMB_SZ)) 
        db = np.concatenate((db, emb_anc.numpy()), axis=0)
        query = np.concatenate((query, emb_pos.numpy()), axis=0)
        
    # Search test
    accs_by_scope, avg_rank_by_scope = eval_mini(query, db, mode='argmax', display=True)
    return


# Main loop
def main():
    if TEST_MODE == 'mini-test-only':
        mini_search_test() # In-memory-search-test...
    elif TEST_MODE == 'save-db':
        generate_emb(n_query=0)
    elif TEST_MODE == 'save-db-query':
        generate_emb(n_query=100000)
    else:
        start_ep = opt.iterations.numpy() // len(train_ds)
        
        for ep in range(start_ep, MAX_EPOCH):
            tr_loss.reset_states()
            ts_loss.reset_states()
            tf.print('EPOCH: {}/{}'.format(ep, MAX_EPOCH))
            progbar = tf.keras.utils.Progbar(len(train_ds))
    
            # Train
            for Xa, Xp in train_ds:
                train_step(Xa, Xp)
                progbar.add(1, values=[("tr loss", tr_loss.result())])
            c_manager.save()  # Save checkpoint...
    
            # Validate
            for Xa, Xp in val_ds:
                val_step(Xa, Xp)
            tf.print('trloss:{:.4f}, tsloss:{:.4f}'.format(tr_loss.result(),
                                                           ts_loss.result()))
            
            # Test 
            if TEST_MODE == 'mini-test':
                mini_search_test() # In-memory-search-test...

    return


if __name__ == "__main__":
    main()

##%%
#train_ds = generator_fp.genUnbalSequence(
#    fns_event_list=music_fps[:8500],
#    bsz=TR_BATCH_SZ,
#    n_anchor= TR_N_ANCHOR, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor 
#    duration=DUR,  # duration in seconds
#    hop=HOP,
#    fs=FS,
#    shuffle=False,
#    random_offset=True,
#    bg_mix_parameter=[True, aug_fps, (10, 10)],
#    ir_mix_parameter=[True, ir_fps])    
#    
#    
#test_ds = generator_fp.genUnbalSequence(
#    music_fps,
#    TS_BATCH_SZ,
#    TS_N_ANCHOR,
#    DUR,
#    HOP,
#    FS,
#    shuffle=True,
#    random_offset=True,
#    bg_mix_parameter=[True, aug_fps, (10, 10)],
#    ir_mix_parameter=[True, ir_fps])    
#    
#    
#    
#import wavio, glob, scipy
#Xa, Xp = test_ds.__getitem__(20)
#for i in range(8):
#    wavio.write('Xa_bak{}.wav'.format(i), Xa[i,0,:], 8000, sampwidth=2)
#
#for i in range(32):
#    wavio.write('Xp_bak{}.wav'.format(i), Xp[i,0,:], 8000, sampwidth=2)
