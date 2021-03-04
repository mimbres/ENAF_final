#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:40:01 2019
python build_db_cpu_new.py exp_v2fix_semihard_320win1_d64_tr100k 4 db ../fingerprint_dataset/music_100k/fma_large_8k/
<AWS>
python build_db_cpu_new.py exp_v2fix_semihard_320win1_d64_tr100k 4 db ../../data/
<AWS, continue from 32542, with bucket_sz=10>
python build_db_cpu_new.py exp_v2fix_semihard_320win1_d64_tr100k 4 db ../../data/ 10 32542
<AWS, generate noisy query with 8000 songs

@author: sunkyun
"""
from joblib import Parallel, delayed
import multiprocessing as mp
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta

import sys, glob, os, uuid, wave
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0" # For CPU, use tf-nightly-cpu=2.1.0
from utils.chunk_data_numpy import chunk_data 
from EATS.networks.kapre2keras.melspectrogram import Melspectrogram
from model.nnfp_l2norm_v2_fixed import FingerPrinter


EXP_NAME = 'exp_v2fix_semihard_320win1_d64_tr100k'
EPOCH = '4'
MODE = 'db'
#DB_SEL = '../fingerprint_dataset/mp3_test/1234/abc/000/'#'100kfull'
BUCKET_SZ = 90 # (default) each npy file will contain 10 songs.
CONTINUE_INDEX_FROM = 0
CONTINUE_INDEX_TO = None
if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
if len(sys.argv) > 2:
    EPOCH = sys.argv[2]
if len(sys.argv) > 3:
    MODE = sys.argv[3]
if len(sys.argv) > 4:
    DB_SEL = sys.argv[4]
if len(sys.argv) > 5:
    BUCKET_SZ = sys.argv[5]
if len(sys.argv) > 6:
    CONTINUE_INDEX_FROM = int(sys.argv[6])
if len(sys.argv) > 7:
    CONTINUE_INDEX_TO = int(sys.argv[7])


SAVED_WEIGHT_DIR = 'logs/checkpoint/' + EXP_NAME + '/'
#OUTPUT_EMB_DIR = 'logs/emb/' + EXP_NAME + '_' + DB_SEL + '/' + str(EPOCH)
N_QUERY = 2000000


# Hyper-parameters
FEAT = 'melspec'  # 'spec' or 'melspec'
FS = 8000
DUR = 1
HOP = .5
EMB_SZ = 64
BSZ = 128

# Directories
if DB_SEL == '100k':
    DATA_ROOT_DIR = '../fingerprint_dataset/music_100k/fma_large_8k/'
    OUTPUT_EMB_DIR = './logs/emb/' + EXP_NAME + '_' + DB_SEL + '/' + str(EPOCH)
elif DB_SEL == '100kfull':
    DATA_ROOT_DIR = '../fingerprint_dataset/music_100k_full/fma_full_8k/'
    OUTPUT_EMB_DIR = './logs/emb/' + EXP_NAME + '_' + DB_SEL + '/' + str(EPOCH)
else:
    DATA_ROOT_DIR = DB_SEL
    OUTPUT_EMB_DIR = './logs/emb/' + EXP_NAME + '_' + datetime.today(
    ).strftime('%y%m%d-%H%M') + '/' + str(EPOCH)
os.makedirs(OUTPUT_EMB_DIR, exist_ok=True)

AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'
TEMP_WAV_DIR = './logs/temp/'


music_fps = list(reversed(glob.glob(DATA_ROOT_DIR + '**/*.[mw][pa][3v]', recursive=True)))
aug_ts_fps = glob.glob(AUG_ROOT_DIR + 'ts/**/*.wav', recursive=True)
ir_fps = glob.glob(IR_ROOT_DIR + '**/*.wav', recursive=True)


# Build & Load model 
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
checkpoint = tf.train.Checkpoint(model=m_fp)
c_manager = tf.train.CheckpointManager(checkpoint, SAVED_WEIGHT_DIR, 3)

if EPOCH == '':
    if c_manager.latest_checkpoint:
        tf.print("-----------Restoring from {}-----------".format(
            c_manager.latest_checkpoint))
        checkpoint.restore(c_manager.latest_checkpoint)
        EPOCH = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]
    else:
        tf.print("-----------No checkpoint to restore!!-----------")
else:    
    checkpoint_fname = SAVED_WEIGHT_DIR + 'ckpt-' + str(EPOCH)
    tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
    checkpoint.restore(checkpoint_fname)
m_fp.trainable = False


# Functions
@tf.function(experimental_relax_shapes=True)
def test_step(X):
    # Pad to keep same batch sz
    X = m_pre(X)
    return m_fp(X) # shape: (BSZ, EMB_SZ)


def batch_test_step(X):
    n_total_frame = X.shape[0]
    n_total_batch = int(np.ceil(X.shape[0] / BSZ))
    idx = np.arange(X.shape[0])
    split_idx = chunk_data(idx, BSZ, 0, pad=True).astype(int)
    preds = []
    for i in range(n_total_batch):
        xb = X[split_idx[i, :], :]
        pred = test_step(xb).numpy()
        if (i+1)*BSZ > n_total_frame:
            n_reduce = (i+1)*BSZ - n_total_frame
            pred = pred[:-n_reduce, :]
        preds.append(pred)
    preds = np.concatenate(preds)
    return preds


def load_audio(fn):
    file_ext = fn[-3:]
    # Convert mp3 to wav, and locate it in temp folder
    if file_ext == 'mp3':
        os.makedirs(TEMP_WAV_DIR, exist_ok=True) # temporary wav files will locate here...
        _fn = TEMP_WAV_DIR + str(uuid.uuid4()) + '.wav'
        _command = 'sox --ignore-length --multi-threaded {} -r 8000 -c 1 -b 16 -V1 {}'.format(fn, _fn)
        os.system(_command)
        fn = _fn
        _remove_wav = True
    else:
        _remove_wav = False
        pass;
    
    # Load audio
    pt_wav = wave.open(fn, 'r')
    _fs = pt_wav.getframerate()
    assert(_fs==FS)
    pt_wav.setpos(0)        
    x = pt_wav.readframes(pt_wav.getnframes())
    x = np.frombuffer(x, dtype=np.int16) / 2**15    
    if _remove_wav:
        os.remove(fn)
    return x.astype(np.float32)


def to_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def print_summary():
    print("exp_name: ", EXP_NAME)
    print("epoch: ", EPOCH)
    print("mode: ", MODE)
    print("db_sel: ", DB_SEL)
    print("output_emb_dir: ", OUTPUT_EMB_DIR)
    return


#%% 
#    progbar = tf.keras.utils.Progbar(len(music_fps))
#    for i, fname in enumerate(list(reversed(music_fps))):
#        progbar.update(i+1)  
""" Song-wise emebedding generator for DB """
def _songwise_gen_emb_db(fname):    
    x = load_audio(fname)
    x_slices = chunk_data(x, int(DUR * FS), int(HOP * FS)) #(N, dur)
    return test_step(tf.constant(x_slices)).numpy()


def save_single_gen_emb_db(audio_fname, output_npy_fpath):
    emb = _songwise_gen_emb_db(audio_fname)
    np.save(output_npy_fpath, emb)
    return


""" Parallel audio-loading, List-wise embedding generator for DB """
def _save_parallel_gen_emb_db(fps, output_npy_fpath, n_core):
    xs = Parallel(n_jobs=n_core,
                 verbose=50)(delayed(load_audio)(_fname) for _fname in fps ) 
    embs = []
    for x in xs:
        x_slices = chunk_data(x, int(DUR * FS), int(HOP * FS)) #(N, dur)
        if x_slices.shape[0] > 0:        
            embs.append(batch_test_step(x_slices))
        else:
            pass;
    
    if embs!=[]:
        embs = np.concatenate(embs) # (NxB, dur)
        np.save(output_npy_fpath, embs)
    return
  
    
def parallel_build_db(fps, output_npy_root=OUTPUT_EMB_DIR, bucket_sz=2,
                      n_core=max(2, mp.cpu_count()-2), index_start=0, index_end=None):
    """
    bucket_sz=1000: 1000 songs(5min) = about 150Mb
    """
    os.makedirs(output_npy_root, exist_ok=True)
    divided_fps = list(to_chunks(fps, bucket_sz))
    if index_end==None:
        index_end = len(divided_fps)
    
    progbar = tf.keras.utils.Progbar(len(divided_fps))
    
    for i, fps_part in enumerate(divided_fps):
        # (Option) Continue from previous inference job...
        if i < index_start:
            continue;
        elif i > index_end:
            break;

        start = timer()
        
        save_file_path = output_npy_root + '/db_part{:07d}.npy'.format(i)
        _save_parallel_gen_emb_db(fps_part, save_file_path, n_core)
        progbar.update(i+1)
        
        end = timer()
        print('----', timedelta(seconds=end-start))
    return


#collect_npy_to_memmap(source_dir='/ssd1/skt_50M_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4')
def collect_npy_to_memmap(source_dir, target_dir=None):
    # source_dir = '/ssd1/skt_50M_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4'
    # target_dir = '/ssd1/skt_50M_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4_memmap'
    if target_dir==None:
        target_dir = source_dir + '_memmap/'
    
    all_files = glob.glob(source_dir + '**/*.npy', recursive=True)
    os.makedirs(target_dir, exist_ok=True)

    # Create memmap
    mm = np.memmap(target_dir + '/db.mm', dtype='float32', mode='w+', shape=(1,1))
    del(mm); # close memmap
    
    progbar = tf.keras.utils.Progbar(len(all_files))
    db_shape = [0, None] # (n, dim)
    for l, source_file in enumerate(all_files):
        # Load partial db
        _dt = np.load(source_file)
        
        # Store old DB length, and Update new DB shape
        _old_len_db = db_shape[0]
        db_shape[0] = db_shape[0] + _dt.shape[0]
        db_shape[1] = _dt.shape[1]
        _new_len_db = db_shape[0]
        # Increase memmap DB size
        mm = np.memmap(target_dir + '/db.mm', dtype='float32', mode='r+', shape=tuple(db_shape))        
        
        # Update memmap DB
        mm[_old_len_db:_new_len_db, :] = _dt
        mm.flush(); del(mm); # Confirm update, then close.
        progbar.update(l+1)

    # Save shape
    np.save(target_dir + '/db_shape.npy', db_shape)
    return


def concat_memmap(input_dir1, input_dir2, output_dir=None):
    # input_dir1 = 'logs/emb/exp_v2fix_semihard_320win1_d64_tr100k_100kfull/4'
    # input_dir1 = 'logs/emb/exp_v2fix_semihard_320win1_d64_tr100k_skt_query/4'
    # input_dir2 = '/ssd1/skt_500k_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4_memmap_fma_skt_fmaquery'
    # input_dir2 = '/ssd1/skt_500k_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4_memmap_fma_skt_sktquery'

    input1_shape = np.load(input_dir1 + '/db_shape.npy') # db matched with query
    input2_shape = np.load(input_dir2 + '/db_shape.old') # db with no query
    output_shape = [input1_shape[0] + input2_shape[0], input1_shape[1]]
    
    if output_dir==None:
        output_dir = input_dir2
    output_file = output_dir + '/db_concat.mm'
    
    mm1 = np.memmap(input_dir1 + '/db.mm', dtype='float32', mode='r', shape=tuple(input1_shape)) 
    mm2 = np.memmap(input_dir2 + '/db.old', dtype='float32', mode='r', shape=tuple(input2_shape)) 
    mm_output = np.memmap(output_file, dtype='float32', mode='w+', shape=tuple(output_shape))
    
    mm_output[:len(mm1), :] = mm1
    mm_output[len(mm1):, :] = mm2
    mm_output.flush()
    
    del(mm1); del(mm2); del(mm_output);
    print('Concatenated output DB is saved as {}'.format(output_file))
    
    np.save(output_dir + '/db_shape_concat.npy', output_shape)
    return




#def songwise_gen_emb_query(fname, offset='random'):
#    """ offset=='random': within +-40 % of hop
#        offset==0.3:      0.3s                 """
#    x = load_audio(fname)
#    if offset=='random':
#        offset_frame = int((np.random.ranf() * 2 - 1) * HOP * 0.4 * FS)
#    else:
#        offset_frame = int(offset * FS)
#    
#    if offset_frame >= 0:
#        start_frame = offset_frame
#    else:
#        start_frame = int(HOP * FS + offset_frame)
#    
#    x_slices = chunk_data(x[start_frame:], int(DUR * FS), int(HOP * FS))
#    
#    return
            



        
    
        
    # divide filepath list
#    fps_job_list = list(to_chunks(fps, bucket_sz))
#    aa = Parallel(
#        n_jobs=n_core,
#        verbose=50)(delayed(save_bucket_gen_emb_db)(i, _fps)
#                    for (i, _fps) in enumerate(fps_job_list))
    
#my_list = range(10)
#squares = []
#def find_square(i):
#    return i ** 2
#
#for index, element in enumerate(my_list):
#    squares.append(find_square(element))
#
#num_cores = 3
#squares = Parallel(n_jobs=num_cores, verbose=50)(delayed(
#    find_square)(i)for i in my_list)

#def save_songwise_gen_query_db(audio_fname, output_npy_fpath):
#    emb = songwise_gen_emb_query(audio_fname)
#    os.makedirs(output_npy_fpath, exist_ok=True)
#    np.save(output_npy_fpath, emb)
#    return

#def a():
#    x, _ = tf.audio.decode_wav(tf.io.read_file(fn))
#    return x
#    
#def b():
#    pt_wav = wave.open(fn, 'r')
#    _fs = pt_wav.getframerate()
#    assert(_fs==FS)
#    pt_wav.setpos(0)        
#    x = pt_wav.readframes(pt_wav.getnframes())
#    x = np.frombuffer(x, dtype=np.int16) / 2**15    
#    return x    
#
#%timeit -n 200 b()   

#def d():
#    start = timer()
#    # single
##    for j in range(500):
##        os.system(_command)
#    
#    # parallel
#    Parallel(n_jobs=4,
#             verbose=50)(delayed(os.system)(_command) for j in range(500) )   
#    end = timer()
#    print('elpased time: ', end-start)
#    return
def main():
    start = timer()

    if 'db' in MODE: 
        parallel_build_db(fps=music_fps, bucket_sz=BUCKET_SZ,
                          index_start=CONTINUE_INDEX_FROM,
                          index_end=CONTINUE_INDEX_TO)
    
    if 'que' in MODE:
        pass;
    
    end = timer()
    print(timedelta(seconds=end-start))
    print_summary()
    return


if __name__ == "__main__":
    main()
    
    

    
