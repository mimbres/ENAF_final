# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Wed Sep 16 22:15:22 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf
from tf_datapipe_utils.tf_record_utils import get_tf_record_ds_from_dir
from tf_datapipe_utils.tdomain_augmentation_ops import input_random_sample, taug_op_bg_mix, taug_op_ir_mix, taug_clipping, taug_downsample_alias

# Path
FP_TF_RECORD_ROOT = '/ssd3/fingerprint_tfrecord'
MUSIC_TRAIN_DIR = f'{FP_TF_RECORD_ROOT}/fma_small/*.tfrecord'
MUSIC_TEST_DIR = f'{FP_TF_RECORD_ROOT}/gtzan/*.tfrecord'
BG_TRAIN_DIR = f'{FP_TF_RECORD_ROOT}/bg_train/*.tfrecord'
BG_VALID_DIR = f'{FP_TF_RECORD_ROOT}/bg_valid/*.tfrecord'
IR_OLD_DIR = f'{FP_TF_RECORD_ROOT}/old_ir/*.tfrecord'

# Parameters
SNR_RANGE = [0, 10] # SNR in dB
IR_FB_RANGE = [.3, .9] # IR feedback
CLIPPING_P = 0.1
CLIPPING_AMP_MAX = 10.



# Get ds, ds_bg, ds_ir
ds = get_tf_record_ds_from_dir(MUSIC_TRAIN_DIR)
ds = ds.map(lambda _x: input_random_sample(x=_x, n_sampled_output=2)) # output: ((T,), (T,)), T=8000

ds_bg = get_tf_record_ds_from_dir(BG_TRAIN_DIR)
ds_bg = ds_bg.map(lambda _x: input_random_sample(x=_x, n_sampled_output=1)) # output: (T,), T=8000
ds_bg = ds_bg.shuffle(100, reshuffle_each_iteration=True).repeat() # bg datasets must be repeated

ds_ir = get_tf_record_ds_from_dir(IR_OLD_DIR)
ds_ir = ds_ir.map(lambda _x: input_random_sample(x=_x, n_sampled_output=1)) 
ds_ir = ds_ir.shuffle(100, reshuffle_each_iteration=True).repeat() # ir datasets must be repeated


# Chain BG & IR with Zipped dataset
ds = tf.data.Dataset.zip((ds, ds_bg)) # ((T,), (T,)), (T,)
ds = ds.map(lambda _x, _xbg: taug_op_bg_mix(x_bypass=_x[0], x=_x[1], xbg=_xbg,
                                            snr_range=SNR_RANGE, rand_master_amp=False)) # output: ((T,), (T,)) for Xa and Xp
ds = tf.data.Dataset.zip((ds, ds_ir))
ds = ds.map(lambda _x, _xir: taug_op_ir_mix(x_bypass=_x[0], x=_x[1], xir=_xir,
                                fb_range=IR_FB_RANGE, rand_master_amp=True)) # output: ((T,), (T,)) for Xa and Xp


# Chain clip & downsample: these aug-methods basically include rand_master_amp=True
# ds = ds.map(lambda _x, _xaug:
#             taug_downsample_alias(x_bypass=_x, x=_xaug, prob=0.25))
ds = ds.map(lambda _x, _xaug:
            taug_clipping(x_bypass=_x, x=_xaug, prob=0.1, amp_max=10.))




    
# TEST
it = iter(ds)
q=next(it)
