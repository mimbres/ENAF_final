# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Thu Jul 23 15:57:43 2020
@author: skchang@cochlear.ai
"""
import os
import tensorflow as tf


def write_tf_record_from_ds(ds=tf.data.Dataset,
                            output_tfrecord_fpath=str(),
                            compression_type='GZIP'):
    """Writing TFRecord file with tf.data.
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/TFRecordWriter
    
        Usage:
            OUTPUT_TF_RECORD_FPATH = 'gs://dlfp-tpu/tf_record_test/test.tfrecord'
            COMPRESSION_TYPE = 'GZIP'
            write_tf_record_from_ds(ds, OUTPUT_TF_RECORD_FPATH, 'GZIP')
            compression_type: "GZIP", "ZLIB", or None (no compression)
    """
    # Serialize to scalar strings (required for TF Recording)
    ds_serialized = ds.map(tf.io.serialize_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE) # <MapDataset shapes: (), types: tf.string>
    ds_serialized = ds_serialized.prefetch(tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(output_tfrecord_fpath,
                                                 compression_type=compression_type)
    writer.write(ds_serialized)
    return

      
# def write_tf_record_from_ds_with_shard(
#                             ds=tf.data.Dataset,
#                             output_tf_record_dir=str(),
#                             compression_type='GZIP',
#                             num_shards=int()):
#     """Writing multiple TFRecord files from a sharded dataset. TFRecord file names
#     will be generated automtically as 'data_000.tfrecord'.
    
#         Usage:
#             OUTPUT_TF_RECORD_DIR = 'gs://dlfp-tpu/tf_record_test/gtzan'
#             COMPRESSION_TYPE = 'GZIP'
#             num_shards = int() or 'auto'. 'auto' will decide optimal number of shards (with 100mb file size)
#             compression_type: "GZIP", "ZLIB", or None (no compression)
#     """
#     os.makedirs(output_tf_record_dir, exist_ok=True) 
#     if num_shards=='auto':
#         """TODO: automatically calculate optmial number of shards"""
#         raise NotImplementedError(num_shards)
    
#     for shard_idx in range(num_shards):
#         ds_sharded = ds.shard(num_shards, shard_idx)
#         output_tfrecord_fpath = output_tf_record_dir + '/data_%d.tfrecord'%(shard_idx)
#         print(f'writing {shard_idx+1}/{num_shards} to {output_tfrecord_fpath}...')
        
#         # Serialize to scalar strings (required for TF Recording)
#         ds_serialized = ds_sharded.map(tf.io.serialize_tensor) # <MapDataset shapes: (), types: tf.string>
#         writer = tf.data.experimental.TFRecordWriter(output_tfrecord_fpath,
#                                                      compression_type=compression_type)
#         writer.write(ds_serialized)
#     return




def get_tf_record_ds(tfrecord_fpath=str(), compression_type='GZIP'):
    """Reading TFRecord file with tf.data.
    https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file
        compression_type: "GZIP", "ZLIB", or None (no compression)
    """
    ds_tfrec = tf.data.TFRecordDataset(tfrecord_fpath, compression_type)
    # Parse serialized scalar strings to Tensor
    ds_tfrec = ds_tfrec.map(lambda _raw_record: tf.io.parse_tensor(_raw_record, tf.float32))
    return ds_tfrec


def get_tf_record_ds_from_dir(tfrecord_dir=str()):
    fps = tf.io.gfile.glob(tfrecord_dir)
    return get_tf_record_ds(fps)


# # Usage

#DATA_ROOT_DIR = 'gs://dlfp-tpu/fingerprint/fingerprint_dataset/music'
#DATA_ROOT_DIR = '../fingerprint_dataset/music'

# OUTPUT_TF_RECORD_FPATH = 'gs://dlfp-tpu/tf_record_test/test.tfrecord'
# COMPRESSION_TYPE = 'GZIP'
# ds_tfrec = get_tf_record_ds(OUTPUT_TF_RECORD_FPATH, COMPRESSION_TYPE)