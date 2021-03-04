# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Wed Oct  7 18:37:56 2020
@author: skchang@cochlear.ai
"""
import wavio, glob, scipy
import tensorflow as tf
import numpy as np
from EATS import generator_fp


# Directories
DATA_ROOT_DIR = '../fingerprint_dataset/music/'
AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'

music_fps = sorted(glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True))
aug_tr_fps = sorted(glob.glob(AUG_ROOT_DIR + 'tr/**/*.wav', recursive=True))
aug_ts_fps = sorted(glob.glob(AUG_ROOT_DIR + 'ts/**/*.wav', recursive=True))
ir_fps = sorted(glob.glob(IR_ROOT_DIR + '**/*.wav', recursive=True))


# Hyper-parameters
FS = 8000
DUR = 5
HOP = 2.5 # max offset margin is 40% of hopsize
TR_BATCH_SZ = 120#240 #320 #80  #40
TR_N_ANCHOR = 60 #120 #64 #16  # 8


# Dataloader
def get_unshuffle_ds():
    ds = generator_fp.genUnbalSequence(
        fns_event_list=music_fps[:8500],
        bsz=TR_BATCH_SZ,
        n_anchor= TR_N_ANCHOR, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor 
        duration=DUR,  # duration in seconds
        hop=HOP,
        fs=FS,
        shuffle=False,
        random_offset=False,
        bg_mix_parameter=[False],#[True, aug_tr_fps, (10, 10)],
        ir_mix_parameter=[False])#[True, ir_fps])
    return ds


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
        bg_mix_parameter=[True, aug_tr_fps, (0, 10)],
        ir_mix_parameter=[True, ir_fps])
    return ds

# Test (unchopped-) sequence data loader, sanity check
ds = get_unshuffle_ds()
x0_org, x0_aug = ds.__getitem__(0)
x1_org, x1_aug = ds.__getitem__(1)
x2_org, x2_aug = ds.__getitem__(2)

np.sum(x0_org[0,:,20000:20010] - x0_org[1,:,0:10])


# Test (unchopped-) sequence data loader, shuffle
ds = get_train_ds()
Xa, Xp = ds.__getitem__(20)
for i in range(16):
    wavio.write('Xa_{}.wav'.format(i), Xa[i,0,:], 8000, sampwidth=2)

for i in range(16):
    wavio.write('Xp_{}.wav'.format(i), Xp[i,0,:], 8000, sampwidth=2)