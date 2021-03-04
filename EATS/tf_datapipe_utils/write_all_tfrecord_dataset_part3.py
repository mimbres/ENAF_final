# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Thu Sep  3 17:03:04 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf
from build_plain_audio_slice_dataset import build_plain_audio_slice_dataset


DATA_SOURCE_ROOT_DIR = 'gs://skchang_dlfp_eu/fingerprint_dataset'
OUTPUT_TFRECORD_ROOT_DIR = 'gs://skchang_dlfp_eu/fingerprint_tfrecord'
# DATA_SOURCE_ROOT_DIR = '/ssd3/fingerprint_dataset'
# OUTPUT_TFRECORD_ROOT_DIR = '/ssd3/fingerprint_tfrecord'


list_source_tag = ['fma_large']

# Record dataset
for i, source_tag in enumerate(list_source_tag):
    build_plain_audio_slice_dataset(source_tag, DATA_SOURCE_ROOT_DIR, 8000,
                                    1., True, True, OUTPUT_TFRECORD_ROOT_DIR)