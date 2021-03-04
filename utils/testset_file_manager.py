# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Wed May  6 16:34:22 2020
@author: skchang@cochlear.ai
"""
import os
import glob
from natsort import natsorted
import wavio
import numpy as np

QUERY_MIREX1000_WAV_ROOT = '../fingerprint_dataset/external_db/mirex1000_8k/mirex1000_query/'

GTZAN_ALL_ROOT = '../fingerprint_dataset/music/GTZAN_8k/'
FMA_SMALL_ROOT = '../fingerprint_dataset/music/fma_small_8k/'
FMA_LARGE_ROOT = '../fingerprint_dataset/music_100k/fma_large_8k/'
FMA_FULL_ROOT = '../fingerprint_dataset/music_100k_full/fma_full_8k/'

OUTPUT_DB_MIREX1000_WAV_ROOT = '../fingerprint_dataset/external_db/mirex1000_8k/mirex1000_db/'

OUTPUT_TXT_ROOT = '../fingerprint_dataset/split_info/'
OUTPUT_TXT_QUERY_MIREX1000_LIST = OUTPUT_TXT_ROOT + 'mirex_ordered_query.txt'
OUTPUT_TXT_DB_MIREX1000_LIST = OUTPUT_TXT_ROOT + 'mirex_ordered_db.txt'
OUTPUT_TXT_GTZAN_ALL_LIST = OUTPUT_TXT_ROOT + 'gtzan_all_random.txt'
OUTPUT_TXT_FMA_SMALL_LIST = OUTPUT_TXT_ROOT + 'fma_small_ordered.txt'
OUTPUT_TXT_FMA_LARGE_LIST = OUTPUT_TXT_ROOT + 'fma_large_ordered.txt'
OUTPUT_TXT_FMA_FULL_LIST = OUTPUT_TXT_ROOT + 'fma_full_ordered.txt'


FS = 8000

# Functions
def check_dir_exist():
    assert(os.path.isdir(QUERY_MIREX1000_WAV_ROOT))
    return

def sorting_lower(flist): # Natural sorting with lower-case of alphabets
    return natsorted([x.lower() for x in flist])

def sorting(flist): # Natural sorting with lower-case of alphabets
    return natsorted(flist)
    
def check_file_exist(fpath):
    assert(os.path.isfile(fpath))
    return

def write_txt_from_list(list_str, fname):
    with open(fname, 'w') as f:
        f.writelines("%s\n" % item for item in list_str)
    return

def cut_and_save_wav(source_fpath, target_fpath, start_sec, length_frame=80248):
    source_audio = wavio.read(source_fpath)
    assert(source_audio.rate == FS)
    start_frame = start_sec * FS
    end_frame = start_frame + length_frame
    
    if len(source_audio.data) < end_frame:
        diff = end_frame - len(source_audio.data)
        assert(diff>0)
        print(f'cut_and_save_wav: padding {diff} frames for {source_fpath} at {start_sec}') 
        cut_data = np.zeros((length_frame,1), dtype=np.int16)
        cut_data[:length_frame-diff,:] = source_audio.data[start_frame:start_frame+length_frame-diff]
    else:
        cut_data = source_audio.data[start_frame:end_frame,:]
    #save
    wavio.write(target_fpath, cut_data, source_audio.rate, sampwidth=source_audio.sampwidth)
    return
    

def map_mirex_db_from_query(fps_query):
    fps_db = [] #
    
    for i, fp_query in enumerate(fps_query):
        # Get query-info 
        _fname = os.path.splitext(os.path.basename(fp_query))[0]
        _sub_fname = str.lower(_fname.split('-')[0]) # ex) 'classical.00074'
        _sub_dir = _sub_fname.split('.')[0] # ex) 'classical'
        _sub_start_sec = int(_fname.split('-')[-1]) # ex) 0 or 10 or 20
        _sub_end_sec = _sub_start_sec + int(_fname.split('-')[-2])
        
        # Find and Copy original files 
        _org_fp = GTZAN_ALL_ROOT + _sub_dir + '/' + _sub_fname + '.wav'
        fp_db = OUTPUT_DB_MIREX1000_WAV_ROOT + _fname + '.wav'   
        cut_and_save_wav(source_fpath=_org_fp, target_fpath=fp_db,
                         start_sec=_sub_start_sec)
        
        # Append to fps_db
        fps_db.append(fp_db)
    print(f'map_mirex_db_from_query: mapped {len(fps_query)} MIREX query files to' + 
          f' {len(fps_db)} GTZAN db files!!')
    return fps_db
#%%

def gen_split_test_filelists(output_txt_dir=OUTPUT_TXT_ROOT):
    os.makedirs(OUTPUT_TXT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DB_MIREX1000_WAV_ROOT, exist_ok=True)
    """QUERY_MIREX1000"""
    _fps_q = sorting_lower(glob.glob(QUERY_MIREX1000_WAV_ROOT + '**/*.wav', recursive=True))
    _fps_db = map_mirex_db_from_query(_fps_q)
    write_txt_from_list(_fps_q, OUTPUT_TXT_QUERY_MIREX1000_LIST)
    write_txt_from_list(_fps_db, OUTPUT_TXT_DB_MIREX1000_LIST)
    
    return


def get_fns_from_txt(txt_path):
    fns = []
    with open(txt_path, 'r') as f:
        fns = f.read().splitlines()
    return fns