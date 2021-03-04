#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:07:10 2019

USAGE:
    CUDA_VISIBLE_DEVICES=0 python train_batch_hard.py <option:EXP_NAME>
    - If EXP_NAME is given, it continues training from the latest checkpoint.


@author: skchang@cochelar.ai
"""
import sys, glob, random
import tensorflow as tf
from datetime import datetime
from tqdm import trange, tqdm
assert sys.version_info >= (3, 5) # Python ≥3.5 required
assert tf.__version__ >= "2.0"

from model.nnfp_l2norm import FingerPrinter
from model.online_triplet import batch_all_triplet_loss, batch_hard_triplet_loss
from dataloader_bpf import build_dataset
from utils.config_gpu_memory_lim import allow_gpu_memory_growth


FEAT = 'melspec' # 'spec' or 'melspec'
LR = 1e-5 #2e-4 
BATCH_SZ = 210#84 # Preferred to be devidable with 7
BATCH_HARD = True # If True:batch_hard_triple_loss; Else:batch_all_triplet_loss 
TRIPLET_MARGIN = 0.5
EPOCHS = 500
TEST_SPLIT = 0.1
SHUFFLE_BUFFER_SZ = 500
DATA_ROOT_DIR = '../fingerprint_dataset/'


if len(sys.argv)>1: 
    EXP_NAME = sys.argv[1]
else:
    EXP_NAME = datetime.now().strftime("%Y%m%d-%H%M")
CHECKPOINT_SAVE_DIR = './logs/checkpoint/{}/'.format(EXP_NAME)
CHECKPOINT_N_HOUR = 1 # None: disable 
LOG_DIR="logs/fit/" + EXP_NAME


# GPU config: This is required if target GPU has smaller vmemory
allow_gpu_memory_growth() # config_gpu_memory_limit(0, 4)


# Prepare, split dataset
fps = glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True) #fps = ['./utils/small_dataset/000140.wav', './utils/small_dataset/777180.wav']
random.seed(1)
random.shuffle(fps)
tr_dataset = build_dataset(fps[:round(len(fps)*(1-TEST_SPLIT))], BATCH_SZ, feat=FEAT)
ts_dataset = build_dataset(fps[round(len(fps)*(1-TEST_SPLIT)):], BATCH_SZ, shuffle=False, feat=FEAT)


# Define Metric & Summary Log
tr_loss = tf.keras.metrics.Mean(name='train_loss')
ts_loss = tf.keras.metrics.Mean(name='test_loss')
tr_fraction = tf.keras.metrics.Mean(name='train_fraction')
tr_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/train')
ts_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/test')


# Build Model
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR,
#                                                           decay_steps=920,
#                                                           decay_rate=0.93,
#                                                           staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=LR)#lr_schedule)
if BATCH_HARD:
    loss_obj = batch_hard_triplet_loss
else:
    loss_obj = batch_all_triplet_loss
model = FingerPrinter() # Input shape:(B,F,T,1)


# Chekcpoint manager
checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
c_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_SAVE_DIR, 3,
                                       CHECKPOINT_N_HOUR)

if c_manager.latest_checkpoint:
    tqdm.write("Restored from {}".format(c_manager.latest_checkpoint))
    checkpoint.restore(c_manager.latest_checkpoint)
else:
  print("Initializing from scratch.")


# Functions
@tf.function
def train_step(x, y): # {x:input segment, y:temporal label}
    with tf.GradientTape() as t:
        emb = model(x)
        # Fraction will be 0 if BATCH_HARD=True 
        loss, fraction = loss_obj(y, emb, TRIPLET_MARGIN)
        with tr_summary_writer.as_default():
            tf.summary.scalar('loss', tr_loss.result(), step=opt.iterations)
            tf.summary.scalar('fraction', tr_fraction(fraction), step=opt.iterations)    
    g = t.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    tr_loss(loss)
    tr_fraction(fraction)
    return

@tf.function
def test_step(x, y):
    emb = model(x)
    loss, _ = loss_obj(y, emb, TRIPLET_MARGIN)
    ts_loss(loss)
    with ts_summary_writer.as_default():
        tf.summary.scalar('loss', ts_loss.result(), step=opt.iterations)
    return



# Main
def main():
    for ep in trange(0, EPOCHS, desc='epochs', position=0, ascii=True):
        tr_loss.reset_states()
        ts_loss.reset_states()

        for x, y in tqdm(tr_dataset.shuffle(SHUFFLE_BUFFER_SZ),
                         desc='tr_step', position=1, ascii=True):
            train_step(x, y)
        
        for x, y in tqdm(ts_dataset, desc='ts_step', position=2, ascii=True):
            test_step(x, y)
            
        msg = 'ep:{}, train_loss:{:.4f}, train_fraction:{:.2f}, test_loss:{:.4f}'
        tqdm.write(msg.format(ep, tr_loss.result(), tr_fraction.result(), ts_loss.result()))
        c_manager.save() # Save checkpoint...
    return


if __name__== "__main__":
  main()



def test_code():
    return

