# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Tue Jul 08 15:14:32 2020
@author: skchang@cochlear.ai

USAGE:
    CUDA_VISIBLE_DEVICES=0 python train_xxx.py <opt1:(str)EXP_NAME> <opt2:(int)EPOCH> <opt3:(str)TEST_MODE>

OPTIONS:
    - If <opt1:EXP_NAME> is given and pretrained model exists:
        - If <opt2> is empty, it continues training from the latest checkpoint.
        - Else if <opt2> is given, it continues training from the <opt2:EPOCH>.
    - If <opt1:EXP_NAME> is given and pretrained model doesn't exist, fresh start training.
    - If <opt1:EXP_NAME> is not given, EXP_NAME will be generated by dd/hh/mm/ss.
    - If <opt1>, <opt2> and <opt3:TEST_MODE> are given:
        - If <opt3:TEST_MODE> == "mini-test-only", it loads pre-trained model and proceed in-memory-search-mini-test, then quit.
        - If <opt3:TEST_MODE> == "save-db", it loads pre-trained model and only saves embeddings for DB as a numpy file.
        - If <opt3:TEST_MODE> == "save-db-query", it loads pre-trained model and saves embeddings for DB and Query as numpy files.
LOG-DIRECTORIES:
    
python train_with_EATS_v2fix_batch_NTxent_simCLR_5120win1_noaug_TPU_clip3_lr08.py exp_NTxent_simCLR_5120_noaug_TPU_clip3_lr08
python train_with_EATS_v2fix_batch_NTxent_simCLR_5120win1_noaug_TPU_clip3_lr08.py exp_NTxent_simCLR_5120_noaug_TPU_clip3_lr08 '' mini-test-only
CUDA_VISIBLE_DEVICES=0 train_with_EATS_v2fix_batch_NTxent_simCLR_640win1_server_dm.py exp_NTxent_simCLR_5120_noaug_TPU_clip3_lr08 '' save-db-query 100k 
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_5120_noaug_TPU_clip3_lr08/61/100k 2000 ivfpq-rr 20

@author: skchang@cochlear.ai
"""
#%%
import sys, glob, os, time
import numpy as np
import tensorflow as tf
from datetime import datetime
from EATS import generator_fp
from EATS.networks.kapre2keras.melspectrogram import Melspectrogram
from utils.plotter import save_imshow, get_imshow_image
from utils.config_gpu_memory_lim import allow_gpu_memory_growth, config_gpu_memory_limit
from utils.TPU_init_util import init_tpu # for TPU
#from model.nnfp_l2norm_v2_fixed import FingerPrinter
from model.nnfp_dm import FingerPrinter
from model.online_NTxent_variant_loss_tpu import OnlineNTxentVariantLoss
#from model.online_NTxent_variant_loss import OnlineNTxentVariantLoss
#from model.online_triplet_v2_fixed import Online_Batch_Triplet_Loss
from utils.eval_metric import eval_mini 
from model.lamb_optimizer import LAMB
#import tensorflow_addons as tfa
from model.spec_augments.specaug_chain import SpecAugChainer


# TPU setup
USE_TPU = True
TPU_NAME = 'skchang-tpu-fp-v3-8d'
N_REPLICAS = 8 # 1) num of replicas = num of cores, 2) batch_sz should be dividable with num of replicas  
GS_STORAGE_PREFIX = f'gs://dlfp-tpu/{TPU_NAME}' #'gs://dlfp-tpu-europe-west4'

# Specaug parameters
SPECAUG_CHAIN = ['cutout', 'horizontal']
SPECAUG_PROBS = 0.5
SPECAUG_N_HOLES = 1
SPECAUG_HOLE_FILL = 'zeros'

# Generating Embedding 
GEN_EMB_DATASEL = '10k' # '10k' '100k_full'

# NTxent-parameters
TAU = 0.05 # 0.1 #1. #0.3
REP_ORG_WEIGHTS = 1.0
SAVE_IMG = False
BN = 'layer_norm2d' #'batch_norm'
OPTIMIZER = LAMB #tfa.optimizers.LAMB # tf.keras.optimizers.Adam # LAMB
LR = 8e-4#1e-4#5e-5#3e-5  #2e-5
LABEL_SMOOTH = 0
LR_SCHEDULE = ''
GRAD_CLIP_NORM = 3.

# Hyper-parameters
FEAT = 'melspec'  # 'spec' or 'melspec'
FS = 8000
DUR = 1
HOP = .5
EMB_SZ = 128  #256 not-working now..
TR_BATCH_SZ = 5120 # 640#320 #120#240 #320 #80  #40
TR_N_ANCHOR = 2560 #320#160 #60 #120 #64 #16  # 8
VAL_BATCH_SZ = 128 #120
VAL_N_ANCHOR = 64 #60
TS_BATCH_SZ = 320 #160 #160
TS_N_ANCHOR = 160 #32 #80
MAX_EPOCH = 1000
EPOCH = ''
TEST_MODE = ''

# Directories
DATA_ROOT_DIR = '../fingerprint_dataset/music/'
AUG_ROOT_DIR = '../fingerprint_dataset/aug/'
IR_ROOT_DIR = '../fingerprint_dataset/ir/'

music_fps = sorted(glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True))
aug_tr_fps = sorted(glob.glob(AUG_ROOT_DIR + 'tr/**/*.wav', recursive=True))
aug_ts_fps = sorted(glob.glob(AUG_ROOT_DIR + 'ts/**/*.wav', recursive=True))
ir_fps = sorted(glob.glob(IR_ROOT_DIR + '**/*.wav', recursive=True))


# Setup TPU
if USE_TPU:
    strategy = init_tpu(tpu_name=TPU_NAME, n_replicas=N_REPLICAS)
else:
    allow_gpu_memory_growth()  # GPU config: This is required if target GPU has smaller vmemory
    strategy = None
#%%
#EXP_NAME = 'exp_NTxent_simCLR_use_anc_rep_120_tau0p05'
#EPOCH = 39
TEST_MODE = 'mini-test'


if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
if len(sys.argv) > 2:
    EPOCH = sys.argv[2]
if len(sys.argv) == 1:
    EXP_NAME = datetime.now().strftime("%Y%m%d-%H%M")
if len(sys.argv) > 3:
    TEST_MODE = sys.argv[3]
if len(sys.argv) > 4:
    GEN_EMB_DATASEL = sys.argv[4]
#%%
CHECKPOINT_SAVE_DIR = f'{GS_STORAGE_PREFIX}/logs/checkpoint/{EXP_NAME}/'
CHECKPOINT_N_HOUR = 1  # None: disable
LOG_DIR = f'{GS_STORAGE_PREFIX}/logs/fit/{EXP_NAME}'
IMG_DIR = f'{GS_STORAGE_PREFIX}/logs/images/{EXP_NAME}'
EMB_DIR = f'{GS_STORAGE_PREFIX}/logs/emb/{EXP_NAME}'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Dataloader
def get_train_ds():
    ds = generator_fp.genUnbalSequence(
        fns_event_list=music_fps[:-1000],
        bsz=TR_BATCH_SZ,
        n_anchor= TR_N_ANCHOR, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor 
        duration=DUR,  # duration in seconds
        hop=HOP,
        fs=FS,
        shuffle=True,
        random_offset=True,
        bg_mix_parameter=[True, aug_tr_fps, (10, 10)],
        ir_mix_parameter=[True, ir_fps])
    return ds

def get_val_ds():
    ds = generator_fp.genUnbalSequence(
        music_fps[-1000:],
        VAL_BATCH_SZ,
        VAL_N_ANCHOR,
        DUR,
        HOP,
        FS,
        shuffle=False,
        random_offset=True,
        bg_mix_parameter=[True, aug_ts_fps, (10, 10)],
        ir_mix_parameter=[True, ir_fps])
    return ds

def get_test_ds():
    if GEN_EMB_DATASEL == '100k':
        test_dir = '../fingerprint_dataset/music_100k/'
    elif GEN_EMB_DATASEL == '100k_full':
        test_dir = '../fingerprint_dataset/music_100k_full/'
    elif GEN_EMB_DATASEL == '10k':
        test_dir = '../fingerprint_dataset/music/'
    
    test_fps = sorted(glob.glob(test_dir + '**/*.wav', recursive=True))
    if GEN_EMB_DATASEL == '10k':
        test_fps = test_fps[1000:]
    
    ds = generator_fp.genUnbalSequence(
        list(reversed(test_fps)), # reversed list for later on evaluation!!
        TS_BATCH_SZ,
        TS_N_ANCHOR,
        DUR,
        HOP,
        FS,
        shuffle=False,
        random_offset=False,
        bg_mix_parameter=[True, aug_ts_fps, (10, 10)],
        ir_mix_parameter=[True, ir_fps])
    return ds

# Define Metric, Summary Log
with strategy.scope():
    tr_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    tr_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/train')
    val_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/val')
    ts_summary_writer = tf.summary.create_file_writer(LOG_DIR + '/mini_test')
    ts_summary_writer_dict = dict()
    for key in ['gf', 'f', 'f_postL2']: 
        ts_summary_writer_dict[key] = tf.summary.create_file_writer(LOG_DIR + '/mini_test/' + key)
    image_writer = tf.summary.create_file_writer(LOG_DIR + '/images')
    

# Build Model: m_pre, m_fp, m_specaug
with strategy.scope():
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
    m_specaug = SpecAugChainer(chain_config=SPECAUG_CHAIN, probs=SPECAUG_PROBS,
                                n_holes=SPECAUG_N_HOLES, hole_fill=SPECAUG_HOLE_FILL)
    assert(m_specaug.bypass==False)
    m_specaug.trainable = False
    
    m_fp = FingerPrinter(emb_sz=EMB_SZ, fc_unit_dim=[32, 1], norm=BN, use_L2layer=True)


# Define Optimizer & Loss
with strategy.scope():
    if LR_SCHEDULE == 'cos':
        lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=LR,
            decay_steps=len(get_train_ds()) * MAX_EPOCH, alpha=1e-07)
        opt = OPTIMIZER(learning_rate=lr_schedule, clipnorm=GRAD_CLIP_NORM)
    elif LR_SCHEDULE == 'cos-restart':
        lr_schedule = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=LR,
            first_decay_steps=len(int(get_train_ds())*0.1), num_periods=0.5, alpha=2e-07)
        opt = OPTIMIZER(learning_rate=lr_schedule, clipnorm=GRAD_CLIP_NORM)
    else:
        opt = OPTIMIZER(learning_rate=LR)

    TR_N_REP = TR_BATCH_SZ - TR_N_ANCHOR
    assert(TR_N_REP==TR_N_ANCHOR)
    Online_NTxent_loss_tr = OnlineNTxentVariantLoss(
        local_bsz=TR_BATCH_SZ//N_REPLICAS, tau=TAU, LARGE_NUM=1e11) # 1e9
    loss_obj_tr = Online_NTxent_loss_tr.loss_fn
    
    VAL_N_REP = VAL_BATCH_SZ - VAL_N_ANCHOR
    assert(VAL_N_REP==VAL_N_ANCHOR)
    Online_NTxent_loss_val = OnlineNTxentVariantLoss(
        local_bsz=VAL_BATCH_SZ//N_REPLICAS, tau=TAU, LARGE_NUM=1e11) # 1e9
    loss_obj_val = Online_NTxent_loss_val.loss_fn


with strategy.scope():
    # Chekcpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=m_fp)
    c_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_SAVE_DIR, 3,
                                           CHECKPOINT_N_HOUR)
    
    if EPOCH == '':
        if c_manager.latest_checkpoint:
            print("-----------Restoring from {}-----------".format(
                c_manager.latest_checkpoint))
            checkpoint.restore(c_manager.latest_checkpoint)
            EPOCH = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]
        else:
            print("-----------Initializing from scratch-----------")
    else:    
        checkpoint_fname = CHECKPOINT_SAVE_DIR + 'ckpt-' + str(EPOCH)
        print("-----------Restoring from {}-----------".format(checkpoint_fname))
        checkpoint.restore(checkpoint_fname)
    



# Trainer
@tf.function
def train_step(Xs):
    """Train step."""
    feat = m_pre(Xs)  # (nA+nP, F, T, 1)
    #feat = m_specaug(feat)
    m_fp.trainable = True
    with tf.GradientTape() as t:
        emb = m_fp(feat, training=True)  # (B, E)
        loss, sim_mtx, _ = loss_obj_tr(emb, use_tpu=True)
    g = t.gradient(loss, m_fp.trainable_variables)
    opt.apply_gradients(list(zip(g, m_fp.trainable_variables)))
    
    # Logging
    tr_loss(loss)
    with tr_summary_writer.as_default():
        tf.summary.scalar('loss', tr_loss.result(), step=opt.iterations)
    return sim_mtx


@tf.function
def val_step(Xs):
    """Validation step."""
    feat = m_pre(Xs)
    m_fp.trainable = False
    emb = m_fp(feat, training=False) 
    loss, sim_mtx, _ = loss_obj_val(emb, use_tpu=True)
    
    # Logging
    val_loss(loss)
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss.result(), step=opt.iterations)
    return sim_mtx


@tf.function
def test_step(Xs, drop_div_enc=False):
    """Test step."""
    feat = m_pre(Xs)
    m_fp.trainable = False
    if drop_div_enc:
        emb = m_fp.front_conv(feat, training=False)  # Use f(x) as embedding:20.04.16.
    else:
        emb = m_fp(feat, training=False) # Use g(f(x))
    return tf.split(emb, [TS_N_ANCHOR, TS_BATCH_SZ - TS_N_ANCHOR], axis=0)  # emb_Anchor, emb_Pos

#%% test functions...
def generate_emb(n_query=3000000):
    """
    Arguemnts:
        n_query: Number of embeddings for queries. If 0, then save only DB.
        data_sel: Selecting dataset {'fma10k', 'fma100k', 'fma100kfull'}
    
    Output Numpy memmap-files:
        logs/emb/<exp_name>/db.mm: (float32) tensor of shape (nFingerPrints, dim)
        logs/emb/<exp_name>/query.mm: (float32) tensor of shape (nFingerprints, nAugments, dim)
        logs/emb/<exp_name>/db_shape.npy: (int) 
        logs/emb/<exp_name>/query_shape.npy: (int)
    
    """
    test_ds = get_test_ds()
    n_augs_per_anchor = int((TS_BATCH_SZ - TS_N_ANCHOR) / TS_N_ANCHOR)
    n_augs_per_batch = n_augs_per_anchor * TS_N_ANCHOR
    n_query = (n_query // n_augs_per_batch) * n_augs_per_batch 
    n_batch_required_for_queries = int(n_query / n_augs_per_batch)
    
    shape_db = (len(test_ds) * TS_N_ANCHOR, EMB_SZ)
    shape_query = (n_query // n_augs_per_anchor, n_augs_per_anchor, EMB_SZ)
    
    # Create memmap, and save shapes
    output_dir = EMB_DIR + '/{}'.format(EPOCH) + '/' + GEN_EMB_DATASEL
    os.makedirs(output_dir, exist_ok=True)
    db = np.memmap(output_dir + '/db.mm', dtype='float32', mode='w+', shape=shape_db)
    np.save(output_dir + '/db_shape.npy', shape_db)
    if (n_query > 0):
        query = np.memmap(output_dir + '/query.mm', dtype='float32', mode='w+', shape=shape_query)
        np.save(output_dir + '/query_shape.npy', shape_query)
    
    progbar = tf.keras.utils.Progbar(len(test_ds))
    """Parallelism to speed up preprocessing (2020-04-15)."""
    enq = tf.keras.utils.OrderedEnqueuer(test_ds, use_multiprocessing=True, shuffle=False)
    enq.start(workers=8, max_queue_size=18)
    # Collect embeddings for full-size on-disk DB search, using np.memmap
    i = 0
    while i < len(enq.sequence):
        progbar.update(i)
        Xa, Xp = next(enq.get()) 
        emb_anc, emb_pos = test_step(Xa, Xp)
        db[i*TS_N_ANCHOR:(i+1)*TS_N_ANCHOR, :] = emb_anc.numpy()
        if (i < n_batch_required_for_queries) & (n_query != 0):
            emb_pos = tf.reshape(emb_pos,
                                 (TS_N_ANCHOR, n_augs_per_anchor, EMB_SZ))  # (B,4,128)
            query[i*TS_N_ANCHOR:(i+1)*TS_N_ANCHOR, :, : ] = emb_pos.numpy()
        i += 1
    enq.stop()
    """End of Parallelism................................."""
    tf.print('------Succesfully saved embeddings to {}-----'.format(output_dir))
    del(db) # close mmap-files
    if (n_query > 0): del(query)
    return


def generate_emb_mirex():
    from utils.testset_file_manager import get_fns_from_txt
    MIREX_DB_INFO = '../fingerprint_dataset/split_info/mirex_ordered_db.txt'
    MIREX_QUERY_INFO = '../fingerprint_dataset/split_info/mirex_ordered_query.txt'
    BSZ_MIREX = 320
    ds_db = generator_fp.genUnbalSequence(get_fns_from_txt(MIREX_DB_INFO),
                                          BSZ_MIREX, BSZ_MIREX, DUR, HOP, FS,
                                          shuffle=False, random_offset=False)
    ds_query = generator_fp.genUnbalSequence(get_fns_from_txt(MIREX_QUERY_INFO),
                                          BSZ_MIREX, BSZ_MIREX, DUR, HOP, FS,
                                          shuffle=False, random_offset=False)
    db, query = np.empty(0), np.empty(0)
    m_fp.trainale = False

    progbar = tf.keras.utils.Progbar(len(ds_db))
    for i, (Xa, _) in enumerate(ds_db):
        progbar.update(i)
        emb = m_fp(m_pre(Xa))
        db = np.concatenate((db, emb.numpy()), axis=0) if db.size else emb.numpy()
    
    progbar = tf.keras.utils.Progbar(len(ds_query))
    for i, (Xa, _) in enumerate(ds_query):
        progbar.update(i)
        emb = m_fp(m_pre(Xa))
        query = np.concatenate((query, emb.numpy()), axis=0) if query.size else emb.numpy()
    
    output_dir = EMB_DIR + '/{}'.format(EPOCH)
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + '/MIREX_db.npy', db)
    np.save(output_dir + '/MIREX_query.npy', query)
    print(f'Succesfully finshed generating embeddings for MIREX DB and Queries...\n')
    return


def mini_search_test(mode='argmin', sel_emb='f', post_L2=False,
                     scopes=[1, 3, 5, 9, 11, 19], save=False, n_samples=3000):
    """Mini search test.
    - mode:'argmin' or 'argmax'
    - sel_emb: 'gf' uses g(f(.)) as embedding. 'f' uses f(.) as embedding.
    """
    test_ds = get_test_ds()
    if sel_emb=='gf':
        drop_div_enc = False
    elif sel_emb=='f':
        drop_div_enc = True
    else:
        raise NotImplementedError(sel_emb)
    
    # Collect mini DB
    db, query = np.empty(0), np.empty(0)
    n_iter = n_samples // TS_BATCH_SZ
    for i in range(n_iter):
        with tf.device("/device:CPU:0"):
            Xa, Xp = test_ds.__getitem__(i)
            Xs = tf.concat((Xa, Xp), axis=0)
            emb_anc, emb_pos = test_step(Xs, drop_div_enc)
        # Post L2 normalize
        if post_L2:
            emb_anc = tf.math.l2_normalize(emb_anc, axis=-1)
            emb_pos = tf.math.l2_normalize(emb_pos, axis=-1)
        emb_pos = tf.reshape(emb_pos, (TS_N_ANCHOR, -1, emb_pos.shape[-1])) # (nAnc, nExam, dEmb)
        db = np.concatenate((db, emb_anc.numpy()), axis=0) if db.size else emb_anc.numpy()
        query = np.concatenate((query, emb_pos.numpy()), axis=0) if query.size else emb_pos.numpy()
    
    # Search test
    accs_by_scope, avg_rank_by_scope = eval_mini(query, db, mode=mode, display=True)
    # Write search test result
    if save:
        key = sel_emb + ('_postL2' if post_L2==True else '')
        with ts_summary_writer_dict[key].as_default():
            for acc, scope in list(zip(accs_by_scope[0], scopes)): # [0] is top1_acc
                tf.summary.scalar(f'acc_{scope}s', acc, step=opt.iterations)
    return


# Main loop
def main():
    if TEST_MODE == 'mini-test-only':
        print('---------ArgMin g(f(x)) mini-TEST----------')
        mini_search_test(sel_emb='gf') # In-memory-search-test...
        print('-----------ArgMin f(x) mini-TEST-----------')
        mini_search_test(sel_emb='f', post_L2=False)
        print('---------ArgMin l2(f(x)) mini-TEST---------')     
        mini_search_test(sel_emb='f', post_L2=True)
    elif TEST_MODE == 'save-db':
        generate_emb(n_query=0)
    elif TEST_MODE == 'save-db-query':
        generate_emb_mirex()
        generate_emb(n_query=3000000)
    else:
        n_iters = len(get_train_ds())
        start_ep = int(opt.iterations) // n_iters
        
        for ep in range(start_ep, MAX_EPOCH):
            tr_loss.reset_states()
            val_loss.reset_states()
            print('EPOCH: {}/{}'.format(ep, MAX_EPOCH))
            progbar = tf.keras.utils.Progbar(n_iters)
    
            # Train
            """Parallelism to speed up preprocessing (2020-04-15)."""
            train_ds = get_train_ds()
            enq = tf.keras.utils.OrderedEnqueuer(train_ds, use_multiprocessing=True, shuffle=train_ds.shuffle)
            enq.start(workers=62, max_queue_size=80)

            i = opt.iterations.numpy() % n_iters # 0 unless using last checkpoint
            progbar.add(i)
            
            #for i in range(len(train_ds)):
            while i < len(enq.sequence):
                i += 1
                Xa, Xp = next(enq.get())
                Xa, Xp = tf.constant(Xa.astype(np.float32)), tf.constant(Xp.astype(np.float32))
                def distribute_value_fn(ctx):
                    rep_id = ctx.replica_id_in_sync_group
                    n_replicas = ctx.num_replicas_in_sync
                    local_n_anc = TR_BATCH_SZ // 2 // n_replicas
                    # _Xa = tf.reshape(Xa, (n_replicas, local_n_anc, 1, -1))
                    # _Xp = tf.reshape(Xp, (n_replicas, local_n_anc, 1, -1))
                    # return tf.concat((_Xa[rep_id,:,:,:], _Xp[rep_id,:,:,:]), axis=0)
                    # BUG FIX 0731
                    _Xa = tf.split(Xa, n_replicas, axis=0)[rep_id]
                    _Xp = tf.split(Xp, n_replicas, axis=0)[rep_id]
                    return tf.concat((_Xa, _Xp), axis=0)

                Xs = strategy.experimental_distribute_values_from_function(distribute_value_fn)
                sim_mtx = strategy.run(train_step, args=(Xs,))
                sim_mtx = tf.concat(sim_mtx.values, axis=0) # from per-replica sim-mtx tensor to normal tensor
                
                progbar.add(1, values=[("tr loss", tr_loss.result())])
                """Required for breaking enqueue."""
            enq.stop()
            """End of Parallelism................................."""
            if SAVE_IMG:
                img_sim_mtx = get_imshow_image(sim_mtx.numpy(), f'Epoch={ep}')
                img_softmax_mtx = get_imshow_image(tf.nn.softmax(sim_mtx,
                    axis=1).numpy(), title=f'Epoch={ep}')
                with image_writer.as_default():
                    tf.summary.image('tr_sim_mtx', img_sim_mtx, step=opt.iterations)
                    tf.summary.image('tr_softmax_mtx', img_softmax_mtx, step=opt.iterations)
            c_manager.save()  # Save checkpoint...
    
            # Validate
            val_ds = get_val_ds()
            enq = tf.keras.utils.OrderedEnqueuer(val_ds, use_multiprocessing=True, shuffle=train_ds.shuffle)
            enq.start(workers=62, max_queue_size=80)
            i = 0
            
            while i < len(enq.sequence):
                i += 1
                Xa, Xp = next(enq.get())
                Xa, Xp = tf.constant(Xa.astype(np.float32)), tf.constant(Xp.astype(np.float32))
                def distribute_value_fn(ctx):
                    rep_id = ctx.replica_id_in_sync_group
                    n_replicas = ctx.num_replicas_in_sync
                    local_n_anc = VAL_BATCH_SZ // 2 // n_replicas
                    # _Xa = tf.reshape(Xa, (n_replicas, local_n_anc, 1, -1))
                    # _Xp = tf.reshape(Xp, (n_replicas, local_n_anc, 1, -1))
                    # return tf.concat((_Xa[rep_id,:,:,:], _Xp[rep_id,:,:,:]), axis=0)
                    # BUG FIX 0731
                    _Xa = tf.split(Xa, n_replicas, axis=0)[rep_id]
                    _Xp = tf.split(Xp, n_replicas, axis=0)[rep_id]
                    return tf.concat((_Xa, _Xp), axis=0)
                
                Xs = strategy.experimental_distribute_values_from_function(distribute_value_fn)
                sim_mtx = strategy.run(val_step, args=(Xs,))
                sim_mtx = tf.concat(sim_mtx.values, axis=0) # from per-replica sim-mtx tensor to normal tensor
            
            enq.stop()
                
            # Save image..
            if SAVE_IMG:
                img_sim_mtx = get_imshow_image(sim_mtx.numpy(), f'Epoch={ep}')
                img_softmax_mtx = get_imshow_image(tf.nn.softmax(sim_mtx,
                    axis=1).numpy(), title=f'Epoch={ep}')
                with image_writer.as_default():
                    tf.summary.image('val_sim_mtx', img_sim_mtx, step=opt.iterations)
                    tf.summary.image('val_softmax_mtx', img_softmax_mtx, step=opt.iterations)
                    
                
            print('trloss:{:.4f}, valloss:{:.4f}'.format(tr_loss.result(),
                                                         val_loss.result()))
            
            # Test 
            if TEST_MODE == 'mini-test':
                print('---------ArgMin g(f(x)) mini-TEST----------')
                mini_search_test(sel_emb='gf', save=True) # In-memory-search-test...
                print('-----------ArgMin f(x) mini-TEST-----------')
                mini_search_test(sel_emb='f', post_L2=False, save=True) # In-memory-search-test...
                print('---------ArgMin l2(f(x)) mini-TEST---------')     
                mini_search_test(sel_emb='f', post_L2=True, save=True) # In-memory-search-test...
    return


if __name__ == "__main__":
    main()
    