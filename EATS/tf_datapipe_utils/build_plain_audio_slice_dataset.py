# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Thu Jul 23 16:02:43 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf
import os
from tf_record_utils import write_tf_record_from_ds


"""Helper func."""
def sec_to_sample(s, sr=8000):
    return int(s * sr)


"""Data_pipeline_utils."""
# Load audio
def load_wav(fpath=str(),
             target_sr=8000):
    """
    returns:
        x: (1xT) tensor audio samples
    """
    x, _sr = tf.audio.decode_wav(tf.io.read_file(fpath), desired_channels=1)
    x = tf.reshape(x, [-1]) # Tx1 --> T
    # Need something like assert(len(x)!=0), assert(target_sr==_sr) here...
    return x # song-wise raw audio 


# Audio slicer
def get_slices(x, filewise_offset_s, slice_win_s, hop_sz_s, pad_end):
    """In most cases, filewise_offset_s=0."""
    filewise_offset_samples = sec_to_sample(filewise_offset_s)
    slice_win_samples = sec_to_sample(slice_win_s)
    hop_sz_samples = sec_to_sample(hop_sz_s)
    x_slices = tf.signal.frame(x[filewise_offset_samples:],
                               slice_win_samples,
                               hop_sz_samples,
                               pad_end=pad_end)
    return tf.data.Dataset.from_tensor_slices(x_slices)


def get_first_slice_only(x, slice_win_s, pad_end):
    slice_win_samples = sec_to_sample(slice_win_s)
    x = tf.signal.frame(x, slice_win_samples, slice_win_samples, pad_end)[0]
    return x


# This dataset generation method is replaceable with TFRecord dataset
def get_ds_from_fps(fps=[],
                    filewise_offset_s=0.,
                    slice_win_s=1.2,
                    hop_sz_s=0.5,
                    pad_end=False,
                    first_seg_only=False,
                    shuffle_file=False,
                    shuffle_seg=False,
                    num_total_shards=0,
                    shard_idx=0
                    ):
    ds = tf.data.Dataset.from_tensor_slices(fps)
    
    # file-wise Sharding 
    if (num_total_shards > 0):
        ds = ds.shard(num_total_shards, shard_idx)
    else:
        pass;
    
    # file-wise shuffle
    if shuffle_file:
        ds = ds.shuffle(8000)
    
    # ds = ds.shuffle()
    ds = ds.map(lambda _fp: load_wav(fpath=_fp, target_sr=8000)) # OUT: song-wise raw audio (T,)
    if first_seg_only: # only used for IR dataset: take first 1sec with padding.
        pad_end = True
        ds = ds.map(lambda _x: get_first_slice_only(x=_x,
                            slice_win_s=slice_win_s, pad_end=pad_end))
                            #slice_win_s=INPUT_LEN_SEC, pad_end=pad_end))
    else:
        ds = ds.flat_map(lambda _song_x: get_slices(x=_song_x,
                            filewise_offset_s=filewise_offset_s,
                            slice_win_s=slice_win_s,#SEG_WIN_SEC,
                            hop_sz_s=hop_sz_s, #SEG_HOP_SEC,
                            pad_end=pad_end)) # OUT: segment (9600,)
    
    # segment-wise shuffle
    if shuffle_seg:
        ds = ds.shuffle()
    return ds 



# Build audio dataset (without augmentation)
def build_plain_audio_slice_dataset(source_tag='gtzan',
                                    data_source_root_dir=str(),
                                    target_sr=8000,
                                    fp_input_segment_length_s=1.,
                                    extended_length_for_tfrecord=True,
                                    write_tfrecord_with_predefined_shard=False,
                                    output_tfrecord_root_dir=None):
    """
    This function builds a plain audio dataset (without augmentation or randomness)
    with extended length for the future use of random offset sampling...
    
    Args:
        source_tag:
            'gtzan':  GTZAN dataset (30s) * 1000 songs
            'fma_small':  FMA 10k (30s) * 9,500 songs
            'fma_medium': FMA 100k (30s) * 100,000 songs
            'fma_large': FMA 100k (full) * 100,000 songs
            'bg_train': background noise samples for train
            'bg_test': background noise sample for test
            'old_ir': (OLD) impulse response samples
            'mic_ir': (NEW) 
            'space_ir':
            
            
        NOTE:
            - 'gtzan', 'fma' or 'ir' tags build a dataset sampled with extended
            length for the use of filewise-offsets.
            - 'bg' tags will build a dataset samples in 'first-seg-only' mode.
    
        data_source_root_dir:
        target_sr: 8000 (default)
        fp_input_segment_length_s: 1s (default)
        
        
        (For writing TF-record, the following options must be set True)
        extended_length_for_tfrecord: exteded sample length
        write_tfrecord_with_predefined_shard: if True, write multiple tfrecords file and returns None         
        output_tf_record_root_dir: (str)
        
    Returns:
        tf.data.Dataset
    """
    max_seg_interval_p = 0.2
    if extended_length_for_tfrecord:
        song_offset_sec = 0
        max_song_offset_sec = 0.25
        seg_win_sec = fp_input_segment_length_s * (1. + max_seg_interval_p + max_song_offset_sec) # 1.45
    else:
        song_offset_sec = 0 # or 0.25
        seg_win_sec = fp_input_segment_length_s * (1. + max_seg_interval_p) # 1.2
        
    seg_hop_sec = fp_input_segment_length_s / 2.
    
    # Get file paths
    first_seg_only = False
    if 'gtzan' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/music/GTZAN_8k/**/*.wav') # 1000
        num_shards = 1        
    elif 'fma_small' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/music/fma_small_8k/**/*.wav') # 7994
        num_shards = 64
    elif 'fma_medium' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/music_100k/fma_large_8k/**/*.wav') # 104579
        num_shards = 800
    elif 'fma_large' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/music_100k_full/fma_full_8k/**/*.wav') # 104632
        num_shards = 8000
    elif 'bg_train' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/aug/tr/**/*.wav') # 1246
        num_shards = 1 
    elif 'bg_valid' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/aug/ts/**/*.wav') # 310
        num_shards = 1 
    elif 'old_ir' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/ir/*.wav') # 243
        seg_win_sec = fp_input_segment_length_s
        first_seg_only = True
        num_shards = 1 
    elif 'mic_ir' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/ir_v2/mic_ir/**/*.wav') # 94
        seg_win_sec = fp_input_segment_length_s
        first_seg_only = True
        num_shards = 1 
    elif 'space_ir' == str.lower(source_tag):
        # The line below is required because gfile.glob doesn't support recursive mode.
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/ir_v2/space_ir/**/*.wav') + \
            tf.io.gfile.glob(f'{data_source_root_dir}/ir_v2/space_ir/**/**/**/**/*.wav') + \
            tf.io.gfile.glob(f'{data_source_root_dir}/ir_v2/space_ir/**/**/**/*.wav') # 780
            
        seg_win_sec = fp_input_segment_length_s
        first_seg_only = True
        num_shards = 1 
    elif 'random_noise' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/random_noise/**/*.wav') # 3600
        seg_win_sec = fp_input_segment_length_s
        first_seg_only = True
        num_shards = 1
    elif 'speech_train' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/speech/common_voice_8k/en/train/*.wav') # 12135 * 4s
        num_shards = 16
    elif 'speech_dev' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/speech/common_voice_8k/en/dev/*.wav') # 7013 * 4s
        num_shards = 1
    elif 'speech_test' == str.lower(source_tag):
        fps = tf.io.gfile.glob(f'{data_source_root_dir}/speech/common_voice_8k/en/test/*.wav') # 7016 * 4s
        num_shards = 1
    else:
        raise NotImplementedError(source_tag)
        
    print(f'Collected {len(fps)} items with source_tag:{source_tag}.')
    print(f'target_sr={target_sr}')
    print(f'fp_input_segment_length_s={fp_input_segment_length_s}')
    print(f'extended_length_for_tfrecord={extended_length_for_tfrecord}')
    print(f'seg_win_sec={seg_win_sec}')
    print(f'seg_hop_sec={seg_hop_sec}')
    print(f'first_seg_only={first_seg_only}')
    
    
    if write_tfrecord_with_predefined_shard:
        # Write tfrecord with predefined number of shards
        progbar = tf.keras.utils.Progbar(num_shards)
        
        for shard_idx in range(num_shards):
            progbar.add(1)
            
            ds_sharded = get_ds_from_fps(
                fps=fps,
                filewise_offset_s=song_offset_sec,
                slice_win_s=seg_win_sec,
                hop_sz_s=seg_hop_sec,
                first_seg_only=first_seg_only,
                num_total_shards=num_shards,
                shard_idx=shard_idx
                )
            
            # Write
            output_tfrecord_dir = output_tfrecord_root_dir + f'/{source_tag}'
            os.makedirs(output_tfrecord_dir, exist_ok=True)
            output_tfrecord_fpath = output_tfrecord_dir + '/data_%d.tfrecord'%(shard_idx)
            print(f'writing {shard_idx+1}/{num_shards} to {output_tfrecord_fpath}...')
            
            write_tf_record_from_ds(ds=ds_sharded,
                            output_tfrecord_fpath=output_tfrecord_fpath,
                            compression_type='GZIP')  
        
        
        return None
    else:
        # Get dataset
        ds = get_ds_from_fps(
            fps=fps,
            filewise_offset_s=song_offset_sec,
            slice_win_s=seg_win_sec,
            hop_sz_s=seg_hop_sec,
            first_seg_only=first_seg_only
            )
        return ds
        
            
            
    
    
