# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Common voice dataset splitter for fingerprint experiment.

Created on Wed Sep  9 14:42:46 2020
@author: skchang@cochlear.ai
"""

import csv
import os
from shutil import copyfile

DATA_ROOT_DIR = '/ssd3/common_voice/downloads/extracted/TAR_GZ.voice-prod-bundl-ee196.s3_cv-corpu-1_enPI94KQNovcg19iBJxE6xLnsSBR1sxy8nA11qzRiYWjI.tar.gz'
TRAIN_TSV = DATA_ROOT_DIR + '/train.tsv'
TEST_TSV = DATA_ROOT_DIR + '/test.tsv'
DEV_TSV = DATA_ROOT_DIR + '/dev.tsv'
OUTPUT_TRAIN_DIR = '/ssd3/fingerprint_dataset/speech/common_voice/en/train'
OUTPUT_TEST_DIR = '/ssd3/fingerprint_dataset/speech/common_voice/en/test'
OUTPUT_DEV_DIR = '/ssd3/fingerprint_dataset/speech/common_voice/en/dev'

def split_files_using_tsv(tsv_fp, output_dir):
    tsv_file = open(tsv_fp)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    
    file_cnt = 0
    os.makedirs(output_dir, exist_ok=True)
    for i, row in enumerate(read_tsv):
        fpath = DATA_ROOT_DIR + '/clips/' + row[1] + '.mp3'
        if os.path.exists(fpath):
            file_cnt += 1
            # copy file
            new_fpath = output_dir + '/' + row[1] + '.mp3'
            copyfile(fpath, new_fpath)
        else:
            print(f'{fpath} does not exist!')
    print(f'Total {file_cnt} files copied')        




def main():
    split_files_using_tsv(TRAIN_TSV, OUTPUT_TRAIN_DIR)
    split_files_using_tsv(TEST_TSV, OUTPUT_TEST_DIR)
    split_files_using_tsv(DEV_TSV, OUTPUT_DEV_DIR)
