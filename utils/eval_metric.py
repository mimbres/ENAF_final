#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:24:48 2019

Main-functions:
    def eval_mini(query, db, scopes, display);
    - mini-test with in-memory-search 
    
    def eval_large();
    - large-scale-test with on-disk-search
    

Sub-functions:
    def _pairwise_distances_for_eval(emb_que, emb_db, return_dotprod, squared)

@author: skchang@cochlear.ai
"""

import tensorflow as tf
import numpy as np

@tf.function
def _pairwise_distances_for_eval(emb_que, emb_db, return_dotprod=False, squared=True):
    """
    Args:
        emb_que: tensor of shape (nQItem, n_augs, d)
        emb_db : tesnor of shape (nDItem, d)
    
    Pairwise L2 squared distance matrix:        
        
      ||a - b||^2 = ||a||^2  + ||b||^2 - - 2 <a, b>
      
    
    Returns:
        dists: tensor of shape (n_augs, nQItem, nDItem, 1)
            
    """
    dot_product = tf.matmul(emb_que, tf.transpose(emb_db)) # (nQItem,n_augs,nDItem)
    dot_product = tf.transpose(dot_product, perm=[1,0,2]) # (n_augs, nQItem, nDItem)
    if return_dotprod:
        return tf.expand_dims(dot_product, 3) #(n_augs, nQItem, nDItem, 1)
    else:
        pass
    
    # Get squared L2 norm for each embedding.
    que_sq = tf.reduce_sum(tf.square(emb_que), axis=2) # (nItem, n_augs)
    que_sq = tf.transpose(que_sq, perm=[1,0]) # (n_augs, nItem)
    db_sq = tf.reduce_sum(tf.square(emb_db), axis=1) # (nItem,)
    db_sq = tf.reshape(db_sq, (1, -1))
    
    dists = tf.expand_dims(que_sq, 2) + tf.expand_dims(db_sq, 1) - 2.0 * dot_product
    dists = tf.maximum(dists, 0.0) # Make sure every dist >= 0
    dists = tf.expand_dims(dists, 3)
    
    if not squared:
    # To prevent divide by 0...
        mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
        dists = dists + mask * 1e-16
        dists = tf.sqrt(dists)
        dists = dists * (1.0 - mask)
        
    return dists


def conv_eye_func(x, s):
    conv_eye = tf.keras.layers.Conv2D(filters=1, kernel_size=[s,s],
                       padding='valid', use_bias=False,
                       kernel_initializer=tf.constant_initializer(np.eye(s).reshape((s,s,1,1))))
    conv_eye.trainable = False
    return conv_eye(x)



def eval_mini(query, db, scopes=[1, 3, 5, 9, 11, 19], mode='argmin', display=True, gt_id_offset=0): 
    """
    mode='argmin': use for minimum distance search
    mode='argmax': use for maximum inner-product search
    """
    n_augs = query.shape[1]
    n_scopes = len(scopes)  

    query = tf.constant(query.astype('float32'))
    db = tf.constant(db.astype('float32'))
    # Compute pair-wise distance matrix, and convolve with eye(scope)     
    if mode=='argmin':
        all_dists = _pairwise_distances_for_eval(query, db, squared=True).numpy()
    elif mode=='argmax':
        all_dists = _pairwise_distances_for_eval(query, db, return_dotprod=True).numpy()
    else:
        raise NotImplementedError(mode)
    
    mean_rank = np.zeros(n_scopes)
    top1_acc, top3_acc, top10_acc = np.zeros(n_scopes), np.zeros(n_scopes), np.zeros(n_scopes)
    for i, s in enumerate(scopes):
#        conv_eye = tf.keras.layers.Conv2D(filters=1, kernel_size=[s,s],
#                       padding='valid', use_bias=False,
#                       kernel_initializer=tf.constant_initializer(np.eye(s).reshape((s,s,1,1))))
#        conv_dists = conv_eye(all_dists).numpy()
        conv_dists = conv_eye_func(all_dists, s).numpy()
        conv_dists = np.squeeze(conv_dists, 3) # (n_augs, n_q, n_db)
        #print(i,s)
        # Mean-rank
        sorted = np.argsort(conv_dists, axis=2)
        if mode=='argmax':
            sorted = sorted[:, :, ::-1]
        n_targets = conv_dists.shape[1]

        _sum_rank = 0
        for target_id in range(n_targets):
            gt_id = target_id + gt_id_offset # this offset is required for large-scale search only
            _, _rank = np.where(sorted[:, target_id, :]==gt_id)
            _sum_rank += np.sum(_rank) / n_augs
        mean_rank[i] = _sum_rank / n_targets
        
        # Top1,Top3,Top10 Acc
        _n_corrects_top1, _n_corrects_top3, _n_corrects_top10 = 0, 0, 0
        for target_id in range(n_targets):
            gt_id = target_id + gt_id_offset # this offset is required for large-scale search only
            _n_corrects_top1 += np.sum(sorted[:, target_id, 0]==gt_id) / n_augs#4
            _n_corrects_top3 += np.sum(sorted[:, target_id, :3]==gt_id) / n_augs#4
            _n_corrects_top10 += np.sum(sorted[:, target_id, :10]==gt_id) / n_augs#4
        top1_acc[i] = _n_corrects_top1 / n_targets
        top3_acc[i] = _n_corrects_top3 / n_targets
        top10_acc[i] = _n_corrects_top10 / n_targets
        
    if display:
        tf.print('scope:  ', scopes)
        text_acc, text_rank = 'T1acc: ', 'mrank: '
        for i, s in enumerate(scopes):
            text_acc += '{:.2f}  '.format(top1_acc[i]*100)
            text_rank += '{:.2f}  '.format(mean_rank[i])
        print(text_acc)
        print(text_rank)
            
    return (top1_acc * 100., top3_acc * 100., top10_acc * 100.), mean_rank


def eval_large(emb_dir=str(), bsz_q=100, scope=[1, 3, 5, 9, 11, 19], mode='argmin'):
    """
    Arguments:
        emb_dir: (str) 'logs/emb/<exp_name>/<epoch>'
        bsz_q: (int) Batch size for query. This must be larger than max scope(19)
        
        logs/emb/<exp_name>/db.mm: (float32) tensor of shape (nFingerPrints, dim)
        logs/emb/<exp_name>/query.mm: (float32) tensor of shape (nFingerprints, nAugments, dim)
        logs/emb/<exp_name>/db_shape.npy: (int) 
        logs/emb/<exp_name>/query_shape.npy: (int)
    
    Variables:
        query: tensor of shape (nQItem, n_augs, d)
        db : tesnor of shape (nDItem, d)
    
    """
    db_fpath = emb_dir + '/db.mm'
    query_fpath = emb_dir + '/query.mm'
    db_shape = np.load(emb_dir + '/db_shape.npy')
    query_shape = np.load(emb_dir + '/query_shape.npy')
    
    # Load DB and queries from memory-mapped-disk
    db = np.memmap(db_fpath, dtype='float32', mode='r', shape=(db_shape[0], db_shape[1])) # (nItem,, dim)
    query = np.memmap(query_fpath, dtype='float32', mode='r',
                      shape=(query_shape[0], query_shape[1], query_shape[2])) # (nItem, nAug, dim)
    
    # Lengths
    #db_length = db.shape[0]
    query_length = query.shape[0]
    #n_augs = query.shape[1]
    n_scopes = len(scope)
    
    
    # Search
    mean_rank = np.zeros(n_scopes)
    top1_acc, top3_acc, top10_acc = np.zeros(n_scopes), np.zeros(n_scopes), np.zeros(n_scopes)
    
    n_mini_test = query_length // bsz_q # number of batches in query
    sum_t1_acc, sum_t3_acc, sum_t10_acc = np.zeros(n_scopes), np.zeros(n_scopes), np.zeros(n_scopes)
    sum_mean_rank = np.zeros(n_scopes)
    
    progbar = tf.keras.utils.Progbar(n_mini_test)
    for i in range(n_mini_test):
        _query = query[i*bsz_q:(i+1)*bsz_q, :, :]
        
        # Perform mini-test with a batch of queries 
        (_t1_acc, _t3_acc, _t10_acc), _mean_rank = eval_mini(query=_query, db=db,
                        scopes=scope, mode=mode, display=True, gt_id_offset=i*bsz_q)
        
        sum_t1_acc += _t1_acc
        sum_t3_acc += _t3_acc
        sum_t10_acc += _t10_acc
        sum_mean_rank += _mean_rank
        # Display Progress
        disp_values = []
        for k, sc in enumerate(scope):
            disp_values.append((str(sc), _t1_acc[k]))
        progbar.add(1, values=disp_values)
    
    top1_acc = sum_t1_acc / n_mini_test
    top3_acc = sum_t3_acc / n_mini_test
    top10_acc = sum_t10_acc / n_mini_test
    mean_rank = sum_mean_rank / n_mini_test
    
    # Display final result
    text_nscope = '\n scope = ' + str(scope)
    text_rank_acc = ' mean_rank = {}\n top_1_acc = {}\n top_3_acc = {}\n top_10_acc = {}'.format(
        mean_rank, top1_acc, top3_acc, top10_acc)
    
    tf.print(text_nscope)
    tf.print(text_rank_acc)
    
    # Save results as 'result.txt' in <emb_dir>
    save_path = emb_dir + '/result.txt'
    file = open(save_path, 'w')
    file.write(text_nscope);
    file.write('\n\n');
    file.write(text_rank_acc);
    file.close();
    tf.print('----------saved evalutation result to {}----------'.format(save_path))
    return